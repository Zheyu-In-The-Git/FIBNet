
import itertools
import importlib
import inspect


import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


import pandas as pd
import numpy as np
import torch.nn.functional as F



# 一些准确率，错误率的方法
def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class BottleneckNets(pl.LightningModule):
    def __init__(self, model_name, encoder, decoder,  utility_discriminator, latent_discriminator, beta = 1.0, lr = 0.0001, **kwargs):


        super(BottleneckNets, self).__init__()

        self.save_hyperparameters()
        self.model_name = model_name

        # 小网络
        self.encoder = encoder # x->z
        self.decoder = decoder # z->u 向量
        self.utility_discriminator = utility_discriminator # u->0/1
        self.latent_discriminator = latent_discriminator

        # 超参数设置
        self.beta = beta
        self.lr = lr
        self.batch_size = kwargs['batch_size']
        self.identity_nums = kwargs['identity_nums']

        # 设置一些激活函数
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # roc的一些参数
        self.roc = torchmetrics.ROC(task='binary')

        # 欧几里得距离
        self.dist = nn.PairwiseDistance(p=2)

        # 打开计算图
        self.example_input_array = torch.randn([self.batch_size, 3, 224, 224])

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # 确实变成了标准差
        std = torch.clamp(std, min=1e-4)
        jitter = 1e-4
        eps = torch.randn_like(std)
        z = mu + eps * (std+jitter)
        return z

    def forward(self, x, u):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.softmax(self.decoder(z, u))
        u_value = self.sigmoid(self.utility_discriminator(u_hat))
        return z, u_hat, mu, log_var, u_value

    def configure_loss(self, pred, true, loss_type):
        if loss_type == 'MSE':
            mse = nn.MSELoss()
            return mse(pred, true)
        elif loss_type == 'BCE':
            bce = nn.BCEWithLogitsLoss()
            return bce(pred, true)
        elif loss_type == 'CE':
            ce = nn.CrossEntropyLoss()
            return ce(pred, true)
        else:
            raise ValueError("Invalid Loss Type!")


    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999

        opt_phi_theta = optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()),lr=self.lr, betas=(b1, b2))
        scheduler_phi_theta = optim.lr_scheduler.StepLR(opt_phi_theta, step_size=20, gamma=0.1)

        opt_eta = optim.Adam(self.latent_discriminator.parameters(), lr = self.lr, betas=(b1, b2))
        scheduler_eta = optim.lr_scheduler.StepLR(opt_eta, step_size=20, gamma=0.1)

        opt_phi = optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(b1, b2))
        scheduler_phi = optim.lr_scheduler.StepLR(opt_phi, step_size=20, gamma=0.1)

        opt_tau = optim.Adam(self.utility_discriminator.parameters(), lr=self.lr, betas=(b1, b2))
        scheduler_tau = optim.lr_scheduler.StepLR(opt_tau, step_size=20, gamma=0.1)

        opt_theta = optim.Adam(self.decoder.parameters(), lr=self.lr, betas=(b1, b2))
        scheduler_theta = optim.lr_scheduler.StepLR(opt_theta, step_size=20, gamma=0.1)

        return [opt_phi_theta, opt_eta, opt_phi, opt_tau, opt_theta], [scheduler_phi_theta, scheduler_eta, scheduler_phi,scheduler_tau, scheduler_theta]

    def loss_fn_KL(self, mu, log_var):
        loss = 0.5 * (torch.pow(mu, 2) + torch.exp(log_var) - log_var - 1).sum(1).mean()
        return loss

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def kl_estimate_value(self, discriminating, act_fn):
        if act_fn == 'Softmax':
            discriminated = self.softmax(discriminating)
            kl_estimate_value = (torch.log(discriminated) - torch.log(1 - discriminated)).sum(1).mean()
            return kl_estimate_value.detach()
        elif act_fn == 'Sigmoid':
            discriminated = self.sigmoid(discriminating)
            kl_estimate_value = (torch.log(discriminated) - torch.log(1-discriminated)).sum(1).mean()
            return kl_estimate_value.detach()
        else:
            raise ValueError("Invalid Loss Type!")


    def calculate_eer(self, metrics, match):
        fpr, tpr, thresholds = self.roc(metrics, match)
        eer = 1.0
        min_dist = 1.0
        for i in range(len(fpr)):
            dist = abs(fpr[i] - (1-tpr[i]))
            if dist < min_dist:
                min_dist = dist
                eer = (fpr[i] + (1-tpr[i])) / 2
        return fpr, tpr, thresholds, eer

    def training_step(self, batch, batch_idx, optimizer_idx):
        ###################
        # data processing #
        ###################
        x, u, _ = batch

        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.decoder(z, u)

        # 从Qz分布中采样的标准正太分布
        q_z = torch.randn_like(mu)

        #########################################
        # training the encoder, utility decoder #
        #########################################
        if optimizer_idx == 0:
            loss_phi_theta = self.configure_loss(u_hat, u, 'CE') + self.beta * self.loss_fn_KL(mu, log_var)
            self.log('KL_divergence', self.loss_fn_KL(mu, log_var), prog_bar=True, logger=True, on_step=True)
            self.log('cross_entropy', self.configure_loss(u_hat, u, 'CE'), prog_bar=True, logger=True, on_step=True)
            self.log('loss_phi_theta', loss_phi_theta, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return {'loss': loss_phi_theta}

        ############################################
        ## training the latent space discriminator##
        ############################################
        if optimizer_idx == 1:

            # 正样本是来自编码器的；负样本来自标准正太分布
            # 正样本
            z_valid = torch.ones(z.size(0), 1)
            z_valid = z_valid.type_as(z)
            z_valid = z_valid.to(torch.float32)

            real_z_discriminator_value = self.latent_discriminator(z.detach())
            real_loss = self.configure_loss(real_z_discriminator_value, z_valid, 'BCE')

            # 负样本
            z_fake = torch.zeros(q_z.size(0), 1)
            z_fake = z_fake.type_as(z)
            z_fake = z_fake.to(torch.float32)

            fake_z_discriminator_value = self.latent_discriminator(q_z.detach())
            fake_loss = self.configure_loss(fake_z_discriminator_value, z_fake, 'BCE')


            loss_eta = (real_loss + fake_loss) * self.beta * 0.5
            self.log('loss_eta', loss_eta, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            return {'loss': loss_eta}


        ##########################################
        ## training the Encoder phi Adversarially#
        ##########################################
        if optimizer_idx == 2:
            z_valid = torch.ones(z.size(0), 1)
            z_valid = z_valid.type_as(z)
            z_valid = z_valid.to(torch.float32)

            real_z_discriminator_value = self.latent_discriminator(z)
            loss_phi = - self.configure_loss(real_z_discriminator_value, z_valid, 'BCE') * self.beta

            self.log('loss_phi', loss_phi, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return {'loss': loss_phi}


        ##############################################
        ## Train Attribute Class Discriminator Omega##
        ##############################################
        if optimizer_idx == 3:

            # 正样本是真实的身份；负样本是表征推断而来的
            u_valid = torch.ones(u.size(0), 1)
            u_valid = u_valid.type_as(u)
            u_valid = u_valid.to(torch.float32)

            u_one_hot = F.one_hot(u, num_classes=self.identity_nums)
            real_u_discriminator_value = self.utility_discriminator(u_one_hot.to(torch.float32))
            loss_real = self.configure_loss(real_u_discriminator_value, u_valid, 'BCE')

            u_fake = torch.zeros(u.size(0), 1)
            u_fake = u_fake.type_as(u)
            u_fake = u_fake.to(torch.float32)

            fake_u_discrimination_value = self.utility_discriminator(u_hat.detach())
            loss_fake = self.configure_loss(fake_u_discrimination_value, u_fake, 'BCE')

            loss_omega = (loss_real + loss_fake) * 0.5
            self.log('loss_omega', loss_omega, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return {'loss': loss_omega}

        ################################################
        # Train the Utility Decoder theta Adversarially#
        ################################################
        if optimizer_idx == 4:
            u_fake = torch.zeros(u.size(0), 1)
            u_fake = u_fake.type_as(u)
            u_fake = u_fake.to(torch.float32)

            fake_u_discrimination_value = self.utility_discriminator(u_hat)
            loss_theta = self.configure_loss(fake_u_discrimination_value, u_fake, 'BCE')
            self.log('loss_theta', loss_theta, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # 识别准确率
            u_accuracy, u_misclass_rate = self.get_stats(self.softmax(self.decoder(z)), u)

            # 记录
            tensorboard_log = {'train_u_accuracy': u_accuracy,
                               'train_u_error_rate': u_misclass_rate}
            self.log_dict(tensorboard_log, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return {'loss':loss_theta}

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        # 数据
        x, u, s = batch

        z, u_hat, mu, log_var, u_value = self.forward(x, u)

        val_loss_phi_theta = self.configure_loss(u_hat, u, 'CE') + self.beta * self.loss_fn_KL(mu, log_var)

        u_accuracy, u_misclass_rate = self.get_stats(u_hat, u)

        tensorboard_logs = {'val_u_accuracy': u_accuracy,
                            'val_u_misclass_rate': u_misclass_rate,
                            'val_loss_phi_theta': val_loss_phi_theta}

        self.log_dict(tensorboard_logs, prog_bar=True, logger=True, on_step=True, on_epoch = True)
        return {'val_loss_phi_theta': val_loss_phi_theta,'val_u_accuracy': u_accuracy}


    def test_step(self, batch, batch_idx):
        # 数据
        img_1, img_2, match = batch
        z_1 = self.encoder(img_1)
        z_2 = self.encoder(img_2)
        return {'z_1':z_1, 'z_2':z_2, 'match':match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)

        bottleneck_net_confusion_cos = {'fpr_cos': fpr_cos, 'tpr_cos': tpr_cos, 'thresholds_coss': thresholds_cos,'eer_cos': eer_cos}
        torch.save(bottleneck_net_confusion_cos, r"C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottlenecknets_confusion_cos.pt")


