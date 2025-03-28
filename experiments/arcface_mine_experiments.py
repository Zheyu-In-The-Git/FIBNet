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
from data import CelebaInterface, LFWInterface, AdienceInterface, CelebaRaceInterface, LFWCasiaInterface
import math
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.manifold import TSNE
import torch.nn.functional as F



torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("Device:", device)


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean





class MineNet(nn.Module):
    def __init__(self, latent_dim, s_dim):
        super(MineNet, self).__init__()
        self.mine_net = nn.Sequential(
            nn.Linear((latent_dim + s_dim), 256),
            torch.nn.ELU(alpha=1.0, inplace=False),
            #torch.nn.ReLU6(),

            nn.Linear(256,128),
            torch.nn.ELU(alpha=1.0, inplace=False),
            #torch.nn.ReLU6(),


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


class ArcfaceMineEstimator(pl.LightningModule):
    def __init__(self, latent_dim, s_dim, pretrained_model):
        super(ArcfaceMineEstimator, self).__init__()
        self.mine_net = MineNet(latent_dim, s_dim)
        self.latent_dim = latent_dim
        self.s_dim = s_dim
        self.model = pretrained_model # arcface 的
        self.model.requires_grad_(False)

    def forward(self, z, s):
        loss = self.mine_net(z, s)
        return loss

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.mine_net.parameters(), lr=0.01, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode="max", factor=0.1, patience=10, min_lr=1e-6,verbose=True,
                                                         threshold=1e-4)
        return {"optimizer": optim_train, "lr_scheduler": scheduler, "monitor": "infor_loss"}

    def training_step(self, batch):
        x, u, s = batch
        _, z = self.model(x,u)
        infor_loss = self.mine_net(z, s)
        self.log('infor_loss', -infor_loss, on_epoch=True, on_step=True, prog_bar=True)
        return infor_loss

def ArcfaceMineMain(model_path, latent_dim, save_name): # savename需要写 模型，特征向量维度，数据集，训练还是测试集
    arcface_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
    pretrained_model = arcface_net.load_from_checkpoint(model_path)

    # 网络模型
    arcfacemineestimator = ArcfaceMineEstimator(latent_dim = latent_dim, s_dim = 1, pretrained_model = pretrained_model)

    # 数据

    '''
    
    #celeba数据
    data_module = CelebaInterface(num_workers=2,
                                  dataset='celeba_data',
                                  batch_size=256,
                                  dim_img=224,
                                  data_dir='D:\datasets\celeba',  # 'D:\datasets\celeba'
                                  sensitive_dim=1,
                                  identity_nums=10177,
                                  sensitive_attr='Male',
                                  pin_memory=False)
    '''



    
    '''
    
    data_module = LFWInterface(num_workers=2,
                               dataset = 'lfw',
                               data_dir='D:\datasets\lfw\lfw112',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='White',
                               purpose='attr_extract',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)
    '''



    '''
    
    data_module = AdienceInterface(num_workers=2,
                               dataset = 'adience',
                               data_dir='D:\datasets\Adience',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='Male',
                               purpose='gender_extract',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)
    



    
    data_module = CelebaRaceInterface(
        num_workers=2,
        dataset='celeba_data',
        batch_size=256,
        dim_img=224,
        data_dir='E:\datasets\celeba',  # 'D:\datasets\celeba'
        sensitive_dim=1,
        identity_nums=10177,
        pin_memory=False
    )

    
    
    data_module = AdienceInterface(num_workers=2,
                                   dataset='adience',
                                   data_dir='D:\datasets\Adience',
                                   batch_size=256,
                                   dim_img=224,
                                   sensitive_attr='Male',
                                   purpose='race_extract',
                                   pin_memory=False,
                                   identity_nums=5749,
                                   sensitive_dim=1)
    '''

    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    lfw_casia_data = LFWCasiaInterface(dim_img=224,
                                       batch_size=256,
                                       dataset='lfw_casia_data',
                                       sensitive_attr='White',
                                       lfw_data_dir=lfw_data_dir,
                                       casia_data_dir=casia_data_dir,
                                       purpose='attr_extract')



    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT',
                                     'lightning_logs/arcface_mine_estimator_race/checkpoints_casialfw/')

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='arcface_mine_estimator_logger')

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

        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=220,
        min_epochs=200,
        logger=logger,
        log_every_n_steps=50,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
    )

    trainer.logger._log_graph = None  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need


    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
    print('Model will be created')
    trainer.fit(arcfacemineestimator, lfw_casia_data, ckpt_path=r'E:\Bottleneck_Nets\experiments\lightning_logs\arcface_mine_estimator_race\checkpoints_casialfw\saved_model\arcface_mine_512\last.ckpt')


if __name__ == '__main__':
    model_path = r'E:\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt'
    ArcfaceMineMain(model_path, latent_dim=512, save_name='arcface_mine_512')