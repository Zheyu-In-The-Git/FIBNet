import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os
import PIL
pl.seed_everything(83)
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
from arcface_resnet50 import ArcfaceResnet50
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.manifold import TSNE
import torch.nn.functional as F

class MineNet(nn.Module):
    def __init__(self, latent_dim, s_dim):
        super(MineNet, self).__init__()
        self.mine_net = nn.Sequential(
            nn.Linear((latent_dim + s_dim), 256),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(256,128),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(128, 1),
        )
    def forward(self, z, s):
        batch_size = z.size(0)
        tiled_z = torch.cat([z, z, ], dim=0) # 按行进行添加数据z
        idx = torch.randperm(batch_size) # 得到batch_size-1的随机整数排列

        shuffled_s = s[idx]
        concat_s = torch.cat([s, shuffled_s], dim=0)
        inputs = torch.cat([tiled_z, concat_s], dim=1)
        logits = self.mine_net(inputs)
        pred_zs = logits[:batch_size]
        pred_z_s = logits[batch_size:]
        loss = -np.log2(np.exp(1)) * (torch.mean(pred_zs) - torch.log(torch.mean(torch.exp(pred_z_s))))
        return loss

class BottleneckMineEstimator(pl.LightningModule):
    def __init__(self, latent_dim, s_dim, pretrained_model):
        super(BottleneckMineEstimator).__init__()





