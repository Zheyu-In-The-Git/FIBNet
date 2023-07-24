import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torchvision.utils
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from model import ResNet50, ArcMarginProduct, FocalLoss
import torchvision.models as models
import pickle
from inception_resnet_v1 import InceptionResnetV1
import itertools

import math

pl.seed_everything(83)
import torch.nn.functional as F
from RAPP_data_interface import CelebaRAPPDatasetInterface

from data.lfw_interface import LFWInterface
from data.adience_interface import AdienceInterface
from experiments.bottleneck_mine_experiments import MineNet
from .RAPP import RAPP, Generator, Discriminator
from arcface_resnet50 import ArcfaceResnet50
from .RAPP import

def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def standardize_tensor(x):
    # 计算每个特征的均值和标准差
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)

    # 对张量进行标准化
    standardized_x = (x - mean) / std

    return standardized_x


def xor(a, b):
    return torch.logical_xor(a, b).int()


def pattern():
    vector_length = 10
    pattern = torch.tensor([0, 1, 0, 1])
    vector = torch.cat([pattern[i % 4].unsqueeze(0) for i in range(vector_length)])
    vector = vector.to(torch.int32)
    return vector




class RAPPMineExperiment(pl.LightningModule):
    def __init__(self, latent_dim, s_dim, patience):
        super(RAPPMineExperiment).__init__()
        self.mine_net = MineNet(latent_dim, s_dim)
        self.latent_dim = latent_dim
        self.s_dim = s_dim
        self.patience = patience

        # 创建RAPP网络 #
        RAPP_model = RAPP()
        RAPP_model = RAPP_model.load_from_checkpoint(r'') # TODO:RAPP的引用路径要写
        self.RAPP_model = RAPP_model
        self.RAPP_model.requires_grad_(False)

        # 生成器
        self.RAPP_Generator_model = self.RAPP_model.generator


        # 人脸匹配器
        self.RAPP_Facematcher_model = self.RAPP_model.face_match

    def forward(self, z, s):
        loss = self.mine_net(z,s)
        return loss


    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.mine_net.parameters(), lr=0.001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode='max', factor=0.1, patience=self.patience, min_lr=1e-8, threshold=1e-4)
        return {'optimizer':optim_train, 'lr_cheduler':scheduler, 'monitor':'infor_loss'}


    def training_step(self, batch):
        x, u, a, s = batch # TODO：数据集要这样设计

        a = a.to(torch.int32)
        c = pattern()
        c = c.to(torch.int32)
        b = xor(a, c)
        b = b.to(torch.int32)


        x_prime = self.RAPP_Generator_model(x, b)

        z = self.RAPP_Facematcher_model(x_prime)

        infor_loss = self.mine_net(z, s)

        self.log('infor_loss', -infor_loss, on_step=True, on_epoch=True, prog_bar=True)
        return infor_loss

# TODO先把RAPP Mine 写完吧
class RAPPMineGender():
    celeba_data_module =



















