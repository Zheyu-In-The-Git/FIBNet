import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os
import PIL
pl.seed_everything(83)
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import BottleneckNets, Encoder, Decoder
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
        super(BottleneckMineEstimator, self).__init__()
        self.mine_net = MineNet(latent_dim, s_dim)
        self.latent_dim = latent_dim
        self.s_dim = s_dim
        self.model = pretrained_model
        self.model.requires_grad_(False)

    def forward(self, z, s):
        loss = self.mine_net(z, s)

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.mine_net.parameters(), lr=0.001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode='max', factor=0.5, patience=5, min_lr=1e-8, threshold=1e-2)
        return {'optimizer': optim_train, 'lr_scheduler':scheduler, 'monitor':'infor_loss'}

    def training_step(self, batch):
        x, u, s =batch
        z, _, _, _ = self.model(x, u)
        infor_loss = self.mine_net(z, s)
        self.log('infor_loss', -infor_loss, on_step=True, on_epoch=True, prog_bar=True)
        return infor_loss

def BottleneckMineMain(arcface_model_path, bottleneck_model_path,latent_dim, beta, save_name): # save_name需要写模型，特征向量维度，数据集，训练还是测试集
    arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.5)
    arcface = arcface_resnet50_net.load_from_checkpoint(arcface_model_path)
    encoder = Encoder(latent_dim = latent_dim, arcface_model=arcface)
    decoder = Decoder(latent_dim=latent_dim, identity_nums=10177, s=64.0, m=0.5, easy_margin=False)
    bottlenecknets = BottleneckNets(model_name='bottleneck', encoder=encoder, decoder=decoder, beta=beta, batch_size=64, identity_nums=10177)
    bottlenecknets_pretrained_model = bottlenecknets.load_from_checkpoint(bottleneck_model_path, encoder=encoder, decoder=decoder)

    # 网络模型
    bottlenecknetsmineestimator = BottleneckMineEstimator(latent_dim=latent_dim, s_dim=1, pretrained_model=bottlenecknets_pretrained_model)

    data_module = CelebaInterface(num_workers=2,
                                  dataset='celeba_data',
                                  batch_size=256,
                                  dim_img=224,
                                  data_dir='D:\datasets\celeba',  # 'D:\datasets\celeba'
                                  sensitive_dim=1,
                                  identity_nums=10177,
                                  sensitive_attr='Male',
                                  pin_memory=False)

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/bottleneck_mine_estimator/checkpoints_beta'+str(beta))

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='bottleneck_mine_estimator_logger_beta' + str(beta))

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="min",
                monitor="infor_loss",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor='infor_loss',
                patience=5,
                mode='min'
            )
        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=100,
        min_epochs=50,
        logger=logger,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
    print('Model will be created')
    trainer.fit(bottlenecknetsmineestimator, data_module)

if __name__ == '__main__':
    arcface_model_path = r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt'
    bottleneck_model_path = r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.01\checkpoints\saved_models\last.ckpt'
    latent_dim = 512
    beta = 0.01
    save_name = 'bottleneck_mine_512_celeba_traindataset'
    BottleneckMineMain(arcface_model_path, bottleneck_model_path, latent_dim, beta, save_name)










