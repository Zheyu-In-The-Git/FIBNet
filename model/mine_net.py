import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

class MineNet(nn.Module):
    def __init__(self, latent_dim, s_dim):
        super(MineNet, self).__init__()
        self.mine_net = nn.Sequential(
            nn.Linear((latent_dim + s_dim), 100),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(100,100),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(100, 100),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(100, 100),
            torch.nn.ELU(alpha=1.0, inplace=False),

            nn.Linear(100, 1),


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
        loss = -np.log2(np.exp(1)) * (torch.mean(pred_zs)) - torch.log(torch.mean(torch.exp(pred_z_s)))
        return loss


class MineEstimator(pl.LightningModule):
    def __init__(self, latent_dim, s_dim, pretrained_model):
        self.mine_net = MineNet(latent_dim, s_dim)
        self.optimizer = torch.optim.Adam(self.mine_net.parameters(), lr=0.01)
        self.latent_dim = latent_dim
        self.s_dim = s_dim
        self.model = pretrained_model
        self.model.requires_grad_(False)

    def training_step(self, batch):
        x, _, s = batch
        z = self.model(x)
        loss = self.mine_net(z, s)
        self.log('loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        info = -loss
        self.log('info', info, on_step=True, on_epoch=True, prog_bar=True)
        return info



