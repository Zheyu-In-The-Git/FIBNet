
import itertools


import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


import pandas as pd
import numpy as np
import torch.nn.functional as F

from model.arcface_models import FocalLoss

# 一些准确率，错误率的方法
def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class BottleneckNets(pl.LightningModule):
    def __init__(self, model_name, arcface_model, encoder, decoder, beta, **kwargs):


        super(BottleneckNets, self).__init__()

        self.save_hyperparameters()
        self.model_name = model_name

        # 把Arcface的参数冻结掉
        self.arcface_model_resnet50 = arcface_model.resnet50
        for name, param in self.arcface_model_resnet50.named_parameters():
            param.requires_grad_(False)

        # 小网络
        self.encoder = encoder # 用预训练 直接就是Resnet50,
        self.decoder = decoder

        # 超参数设置
        self.beta = beta
        self.batch_size = kwargs['batch_size']
        self.identity_nums = kwargs['identity_nums']

        # 设置一些激活函数
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # roc的一些参数
        self.roc = torchmetrics.ROC(task='binary')


    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # 确实变成了标准差
        std = torch.clamp(std, min=1e-4)
        jitter = 1e-4
        eps = torch.randn_like(std)
        z = mu + eps * (std+jitter)
        return z

    def forward(self, x, u):
        x = self.arcface_model_resnet50(x)
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.decoder(z, u)
        return z, u_hat, mu, log_var

    def configure_loss(self, pred, true, loss_type):
        if loss_type == 'MSE':
            mse = nn.MSELoss()
            return mse(pred, true)
        elif loss_type == 'BCE':
            bce = nn.BCEWithLogitsLoss()
            return bce(pred, true)
        elif loss_type == 'CE':
            ce = FocalLoss(gamma=2)
            return ce(pred, true)
        else:
            raise ValueError("Invalid Loss Type!")


    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999

        opt = optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.0001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=3, min_lr=1e-8, threshold=1e-2)

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def loss_fn_KL(self, mu, log_var):
        loss = 0.5 * (torch.pow(mu, 2) + torch.exp(log_var) - log_var - 1).sum(1).mean()
        return loss

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

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
        x, u, _ = batch
        z, u_hat, mu, log_var = self.forward(x, u)
        train_loss = self.configure_loss(u_hat, u, 'CE') + self.beta * self.loss_fn_KL(mu, log_var)

        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('entropy_loss', self.configure_loss(u_hat, u, 'CE'), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('KL_divergence', self.loss_fn_KL(mu, log_var), prog_bar=True, logger=True, on_step=True, on_epoch=True)

        train_acc, train_misclass = self.get_stats(u_hat, u)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_misclass', train_misclass, on_step=True, on_epoch=True, prog_bar=True)

        return train_loss

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        # 数据
        x, u, s = batch

        z, u_hat, mu, log_var = self.forward(x, u)

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
        z_1_mu, z_1_sigma = self.encoder(img_1)
        z_2_mu, z_2_sigma = self.encoder(img_2)

        z_1 = self.sample_z(z_1_mu, z_1_sigma)
        z_2 = self.sample_z(z_2_mu, z_2_sigma)

        return {'z_1':z_1, 'z_2':z_2, 'match':match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)
        self.log('eer_cos', eer_cos, on_epoch=True)
        bottleneck_net_confusion_cos = {'fpr_cos': fpr_cos, 'tpr_cos': tpr_cos, 'thresholds_cos': thresholds_cos,'eer_cos': eer_cos}

        torch.save(bottleneck_net_confusion_cos, r'lightning_logs/bottleneck'+str(self.beta)+'.pt')


