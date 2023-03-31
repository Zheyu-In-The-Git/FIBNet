
import itertools
import importlib
import inspect


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, model_checkpoint


import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter


import pandas as pd
import numpy as np
import torch.nn.functional as F



# 一些准确率，错误率的方法
def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class BottleneckNets(pl.LightningModule):
    def __init__(self, model_name, encoder, decoder, uncertainty_model, sensitive_discriminator, utility_discriminator, lam = 0.0001, gamma = 0.0001, lr = 0.0001, **kwargs):
        super(BottleneckNets, self).__init__()

        self.save_hyperparameters()
        self.model_name = model_name

        # 小网络
        self.encoder = encoder # x->z
        self.decoder = decoder # z->u 向量
        self.uncertainty_decoder = uncertainty_model # z->s 向量 要选最大的一个
        self.sensitive_discriminator = sensitive_discriminator # s->0/1
        self.utility_discriminator = utility_discriminator # u->0/1

        # 超参数设置
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.batch_size = kwargs['batch_size']
        self.identity_nums = kwargs['identity_nums']

        # 设置一些激活函数
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()


        # 监督训练结果

        # 打开计算图
        self.example_input_array = torch.randn([self.batch_size, 3, 224, 224])


    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # 确实变成了标准差
        std = torch.clamp(std, min = 1e-4)
        jitter = 1e-4
        eps = torch.randn_like(std)
        z = mu + eps * (std+jitter)

        return z

    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.softmax(self.decoder(z))
        s_hat = self.sigmoid(self.uncertainty_decoder(z))

        u_value = self.sigmoid(self.utility_discriminator(u_hat))
        s_value = self.sigmoid(self.sensitive_discriminator(s_hat))

        return z, u_hat, s_hat, u_value, s_value, mu, log_var

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

        opt_phi_theta_xi = optim.Adam(itertools.chain(self.encoder.parameters(),
                                                      self.decoder.parameters(),
                                                      self.uncertainty_decoder.parameters()),
                                                      lr=self.lr, betas=(b1, b2))

        opt_omiga = optim.Adam(self.utility_discriminator.parameters(), lr = self.lr, betas=(b1, b2))

        opt_tao = optim.Adam(self.sensitive_discriminator.parameters(), lr=self.lr, betas=(b1, b2))


        return [opt_phi_theta_xi, opt_omiga, opt_tao, opt_phi_theta_xi], []

    def loss_fn_KL(self, mu, log_var):
        loss = 0.5 * (torch.pow(mu, 2) + torch.exp(log_var) - log_var - 1).sum(1).mean()
        return loss

    def loss_adversarially(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss()
        return loss(y_hat, y)

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def kl_estimate_value(self, discriminating):
        discriminated = self.sigmoid(discriminating) + 1e-4
        kl_estimate_value = (torch.log(discriminated) - torch.log(1-discriminated)).sum(1).mean()
        return kl_estimate_value.detach()

    def training_step(self, batch, batch_idx, optimizer_idx):
        ###################
        # data processing #
        ###################
        x, u, s = batch

        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.decoder(z)
        s_hat = self.uncertainty_decoder(z)

        ################################################################
        # training the encoder, utility decoder and uncertainty decoder#
        ################################################################

        if optimizer_idx == 0:
            loss_phi_theta_xi = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u, 'CE') - self.lam * self.configure_loss(s_hat, s, 'BCE')
            tensorboard_log = {'loss_phi_theta_xi': loss_phi_theta_xi.detach()}
            self.log('KL_divergence', self.loss_fn_KL(mu, log_var), prog_bar=True, logger=True, on_step=True)
            self.log('cross_entropy', self.configure_loss(u_hat, u, 'CE'), prog_bar=True, logger=True, on_step=True)
            self.log('binary_cross_entropy', self.configure_loss(s_hat, s, 'BCE'), prog_bar=True, logger=True, on_step=True)
            self.log('loss_phi_theta_xi', loss_phi_theta_xi, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return  {'loss': loss_phi_theta_xi, 'log':tensorboard_log}

        #################################
        ## training the u_discriminator##
        #################################
        if optimizer_idx == 1:

            u_valid = torch.ones(u.size(0), 1)
            u_valid = u_valid.type_as(u)
            u_valid = u_valid.to(torch.float32)

            u_one_hot = F.one_hot(u, num_classes = self.identity_nums)
            real_u_discriminator_value = self.utility_discriminator(u_one_hot.to(torch.float32))
            loss_real = self.configure_loss(real_u_discriminator_value, u_valid, 'BCE')


            u_fake = torch.zeros(u.size(0), 1)
            u_fake = u_fake.type_as(u)
            u_fake = u_fake.to(torch.float32)

            fake_u_discrimination_value = self.utility_discriminator(u_hat.detach())
            loss_fake = self.configure_loss(fake_u_discrimination_value, u_fake, 'BCE')


            loss_omiga = (loss_real + loss_fake) * self.gamma *0.5
            tensorboard_log = {'loss_omiga':loss_omiga.detach()}
            self.log('loss_omiga',loss_omiga, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return {'loss':loss_omiga, 'log':tensorboard_log}


        #################################
        ## training the s_discriminator##
        #################################
        if optimizer_idx == 2:
            s_valid = torch.ones(s.size(0), 1)
            s_valid = s_valid.type_as(s)
            real_s_discriminator_value = self.sensitive_discriminator(s)
            real_s_loss = self.configure_loss(real_s_discriminator_value, s_valid, 'BCE')

            s_fake = torch.zeros(s.size(0), 1)
            s_fake = s_fake.type_as(s)
            fake_s_discrimination_value = self.sensitive_discriminator(s_hat.detach())
            fake_s_loss = self.configure_loss(fake_s_discrimination_value, s_fake, 'BCE')

            loss_tao = - self.gamma * (real_s_loss + fake_s_loss) * 0.5


            tensorboard_log = {'loss_tao': loss_tao.detach()}
            self.log('loss_tao', loss_tao, prog_bar=True, logger=True, on_step=True, on_epoch = True)


            return {'loss':loss_tao, 'log':tensorboard_log}


        ######################################################################
        ## training the utility decoder and uncertainty decoder adversarially##
        ######################################################################
        if optimizer_idx == 3:
            u_valid = torch.ones(u.size(0), 1)
            u_valid = u_valid.type_as(u)
            u_valid = u_valid.to(torch.float32)

            s_valid = torch.ones(s.size(0), 1)
            s_valid = s_valid.type_as(s)
            s_valid = s_valid.to(torch.float32)


            #对抗损失
            loss_adversarial_phi_theta_xi = self.gamma * self.loss_adversarially(self.utility_discriminator(self.decoder(z)),u_valid) \
                                                 - self.lam * self.loss_adversarially(self.sensitive_discriminator(self.uncertainty_decoder(z)), s_valid)


            # 识别准确率
            u_accuracy, u_misclass_rate = self.get_stats(self.softmax(self.decoder(z)), u)
            s_accuracy, s_misclass_rate = self.get_stats(self.sigmoid(self.uncertainty_decoder(z)), s)


            train_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u, 'CE') - self.lam * self.configure_loss(s_hat, s, 'CE') + \
                               self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

            tensorboard_log = {'loss_adversarial_phi_theta_xi': loss_adversarial_phi_theta_xi.detach(),
                               'train_u_accuracy': u_accuracy,
                               'train_u_error_rate': u_misclass_rate,
                               'train_s_accuracy': s_accuracy,
                               'train_s_error_rate': s_misclass_rate,
                               'train_total_loss': train_loss_total.detach()}

            self.log('loss_adversarial_phi_theta_xi', loss_adversarial_phi_theta_xi, prog_bar=True, logger=True, on_step=True, on_epoch = True)
            self.log_dict(tensorboard_log, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return {'loss':loss_adversarial_phi_theta_xi, 'log': tensorboard_log}

    # 验证步骤
    def validation_step(self, batch, batch_idx):

        # 数据
        x, u, s = batch
        print(batch_idx)

        z, u_hat, s_hat, u_value, s_value, mu, log_var =self.forward(x)


        val_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u,'CE') - self.lam * self.configure_loss(s_hat, s, 'CE') + \
                           self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

        u_accuracy, u_misclass_rate = self.get_stats(u_hat, u)
        s_accuracy, s_misclass_rate = self.get_stats(s_hat, s)

        tensorboard_logs = {'val_loss_total': val_loss_total,
                            'val_u_accuracy': u_accuracy,
                            'val_u_misclass_rate': u_misclass_rate,
                            'val_s_accuracy':s_accuracy,
                            'val_s_misclass_rate': s_misclass_rate
                            }

        self.log_dict(tensorboard_logs, prog_bar=True, logger=True, on_step=True, on_epoch = True)
        return {'val_loss_total': val_loss_total, 'val_u_accuracy': u_accuracy}


    def test_step(self, batch, batch_idx):
        # 数据
        x, u, s = batch

        z, u_hat, s_hat, u_value, s_value, mu, log_var = self.forward(x)

        test_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u,'CE') - self.lam * self.configure_loss(s_hat, s, 'CE') + \
                         self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

        u_accuracy, u_misclass_rate = self.get_stats(u_hat, u)
        s_accuracy, s_misclass_rate = self.get_stats(s_hat, s)

        tensorboard_logs = {'test_loss_total': test_loss_total,
                            'test_u_accuracy': u_accuracy,
                            'test_u_misclass_rate': u_misclass_rate,
                            'test_s_accuracy': s_accuracy,
                            'test_s_misclass_rate': s_misclass_rate
                            }
        self.log_dict(tensorboard_logs, prog_bar=True, logger=True, on_step=True, on_epoch = True)


'''

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
            print('Model', Model)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        # self.model = self.instancialize(Model) # TODO:需要重新思考一下，到底是实例化什么？可能是实例化模型的不同参数，可能做实验可以用


    # TODO：这里可能是工厂模型
    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
'''



