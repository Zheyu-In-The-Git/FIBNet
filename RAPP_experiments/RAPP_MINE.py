import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torchvision.utils
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(83)

from experiments.bottleneck_mine_experiments import MineNet
from .RAPP import RAPP, Generator, Discriminator

from .RAPP_Mine_data_interface import CelebaRAPPMineTrainingDatasetInterface, CelebaRAPPMineTestDatasetInterface, LFWRAPPMineDatasetInterface, AdienceRAPPMineDatasetInterface




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
        RAPP_model = RAPP_model.load_from_checkpoint(os.path.abspath(r'RAPP_experiments/lightning_logs/RAPP_experiments/checkpoints/saved_model/last.ckpt'))
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
        x, u, a, s = batch

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
def RAPPMine(num_workers, dataset_name, batch_size, dim_img, data_dir, identity_nums, sensitive_attr, pin_memory, fast_dev_run):

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/RAPP_Mine_'+ sensitive_attr + '/checkpoints' )

    if dataset_name == 'CelebA_training_dataset':
        data_module_celeba_training = CelebaRAPPMineTrainingDatasetInterface(num_workers, dataset_name, batch_size, dim_img, data_dir, identity_nums, sensitive_attr, pin_memory=pin_memory)
        logger_celeba_train = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='CelebA_training_RAPP_Mine_'+ sensitive_attr + '_logger')
        trainer_celeba_training = pl.Trainer(
            callbacks=[
                ModelCheckpoint(
                    mode='min',
                    monitor='infor_loss',
                    dirpath=os.path.join(CHECKPOINT_PATH, 'CelebA_training_RAPP_Mine_'+ sensitive_attr + '_models'),
                    save_last=True,
                    every_n_train_steps=50
                ),
                LearningRateMonitor('epoch'),
                EarlyStopping(
                    monitor='infor_loss',
                    patience=5,
                    mode='max'
                )
            ],

            default_root_dir=os.path.join(CHECKPOINT_PATH, 'CelebA_training_RAPP_Mine_'+ sensitive_attr + '_models'),
            accelerator='auto',
            devices=1,
            max_epochs=150,
            min_epochs=120,
            logger=logger_celeba_train,
            log_every_n_steps=10,
            precision=32,
            enable_checkpointing=True,
            fast_dev_run=fast_dev_run

        )

        print('CelebA training' + sensitive_attr + ' dataset for RAPP MINE will be testing, the model will be create!')
        model = RAPPMineExperiment(latent_dim=512,s_dim=1, patience=5)
        trainer_celeba_training.fit(model, data_module_celeba_training)

    elif dataset_name == 'CelebA_test_dataset':
        data_module_celeba_test = CelebaRAPPMineTestDatasetInterface(num_workers, dataset_name, batch_size, dim_img, data_dir, identity_nums, sensitive_attr, pin_memory=pin_memory)
        logger_celeba_test = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='CelebA_test_RAPP_Mine_'+ sensitive_attr + '_logger')
        trainer_celeba_test = pl.Trainer(
            callbacks=[
                ModelCheckpoint(
                    mode='min',
                    monitor='infor_loss',
                    dirpath=os.path.join(CHECKPOINT_PATH, 'CelebA_test_RAPP_Mine_' + sensitive_attr + '_models'),
                    save_last=True,
                    every_n_train_steps=50
                ),
                LearningRateMonitor('epoch'),
                EarlyStopping(
                    monitor='infor_loss',
                    patience=5,
                    mode='max'
                )
            ],

            default_root_dir=os.path.join(CHECKPOINT_PATH, 'CelebA_test_RAPP_Mine_' + sensitive_attr + '_models'),
            accelerator='auto',
            devices=1,
            max_epochs=150,
            min_epochs=120,
            logger=logger_celeba_test,
            log_every_n_steps=10,
            precision=32,
            enable_checkpointing=True,
            fast_dev_run=fast_dev_run
        )
        print('CelebA test ' + sensitive_attr + ' dataset for RAPP MINE will be testing, the model will be create!')
        model = RAPPMineExperiment(latent_dim=512, s_dim=1, patience=5)
        trainer_celeba_test.fit(model, data_module_celeba_test)

    elif dataset_name == 'LFW_dataset':
        data_module_lfw = LFWRAPPMineDatasetInterface(num_workers, dataset_name, batch_size, dim_img, data_dir, identity_nums, sensitive_attr, pin_memory=pin_memory)
        logger_lfw = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='LFW_RAPP_Mine' + sensitive_attr + '_logger')
        trainer_lfw = pl.Trainer(
            callbacks=[
                ModelCheckpoint(
                    mode='min',
                    monitor='infor_loss',
                    dirpath=os.path.join(CHECKPOINT_PATH, 'LFW_RAPP_Mine_' + sensitive_attr + '_models'),
                    save_last=True,
                    every_n_train_steps=50
                ),
                LearningRateMonitor('epoch'),
                EarlyStopping(
                    monitor='infor_loss',
                    patience=5,
                    mode='max'
                )
            ],

            default_root_dir=os.path.join(CHECKPOINT_PATH, 'LFW_RAPP_Mine_' + sensitive_attr + '_models'),
            accelerator='auto',
            devices=1,
            max_epochs=500,
            min_epochs=350,
            logger=logger_lfw,
            log_every_n_steps=10,
            precision=32,
            enable_checkpointing=True,
            fast_dev_run=fast_dev_run
        )
        print('LFW ' + sensitive_attr + ' dataset for RAPP MINE will be testing, the model will be create!')
        model = RAPPMineExperiment(latent_dim=512, s_dim=1, patience=15)
        trainer_lfw.fit(model, data_module_lfw)


    elif dataset_name == 'Adience_dataset':
        data_module_adience = AdienceRAPPMineDatasetInterface(num_workers, dataset_name, batch_size, dim_img, data_dir,identity_nums, sensitive_attr, pin_memory=pin_memory)
        logger_adience = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='Adience_RAPP_Mine' + sensitive_attr + '_logger')
        trainer_adience = pl.Trainer(
            callbacks=[
                ModelCheckpoint(
                    mode='min',
                    monitor='infor_loss',
                    dirpath=os.path.join(CHECKPOINT_PATH, 'Adience_RAPP_Mine_' + sensitive_attr + '_models'),
                    save_last=True,
                    every_n_train_steps=50
                ),
                LearningRateMonitor('epoch'),
                EarlyStopping(
                    monitor='infor_loss',
                    patience=5,
                    mode='max'
                )
            ],

            default_root_dir=os.path.join(CHECKPOINT_PATH, 'Adience_RAPP_Mine_' + sensitive_attr + '_models'),
            accelerator='auto',
            devices=1,
            max_epochs=500,
            min_epochs=350,
            logger=logger_adience,
            log_every_n_steps=10,
            precision=32,
            enable_checkpointing=True,
            fast_dev_run=fast_dev_run
        )
        print('LFW ' + sensitive_attr + ' dataset for RAPP MINE will be testing, the model will be create!')
        model = RAPPMineExperiment(latent_dim=512, s_dim=1, patience=15)
        trainer_adience.fit(model, data_module_adience)

    else:
        print('please check the correct datasets name: CelebA_training_dataset, CelebA_test_dataset, LFW_dataset, Adience_dataset')






if __name__ == '__main__':
    celeba_data_dir = 'E:\datasets\celeba'
    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    adience_data_dir = 'E:\datasets\Adience'

    # gender
    RAPPMine(num_workers=0, dataset_name='CelebA_training_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='CelebA_test_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='LFW_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='Adience_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False, fast_dev_run=True)

    # race
    RAPPMine(num_workers=0, dataset_name='CelebA_training_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='CelebA_test_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='LFW_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False, fast_dev_run=True)
    RAPPMine(num_workers=0, dataset_name='Adience_dataset', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False, fast_dev_run=True)



























