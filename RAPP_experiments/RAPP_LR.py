import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger


from .RAPP import RAPP, Generator, Discriminator

from .RAPP_Mine_data_interface import CelebaRAPPMineTrainingDatasetInterface, LFWRAPPMineDatasetInterface, AdienceRAPPMineDatasetInterface


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


class RAPPLogisticRegressionAttack(pl.LightningModule):

    def __init__(self, dataset_name):
        super(RAPPLogisticRegressionAttack, self).__init__()

        self.dataset_name = dataset_name

        self.linear = nn.Linear(512, 2)

        # 创建RAPP 网络
        RAPP_model = RAPP()
        RAPP_model = RAPP_model.load_from_checkpoint(os.path.abspath(r'RAPP_experiments/lightning_logs/RAPP_experiments/checkpoints/saved_model/last.ckpt'))
        self.RAPP_model = RAPP_model
        self.RAPP_model.requires_grad_(False)

        # 生成器
        self.RAPP_Generator_model = self.RAPP_model.generator
        self.RAPP_Generator_model.requires_grad_(False)

        # 人脸匹配器
        self.RAPP_Facematcher_model = self.RAPP_model.face_match
        self.RAPP_Facematcher_model.requires_grad_(False)


    def forward(self, z):
        logits = self.linear(z)
        return logits

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.linear.parameters(), lr=0.001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode='min', factor=0.1, patience=5, min_lr=1e-8)
        return {'optimizer':optim_train, 'lr_scheduler':scheduler, 'monitor':'train_loss'}

    def get_stats(self, logits, labels):
        preds =  torch.argmax(logits, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate


    def training_step(self, batch):
        x, u, a, s = batch

        a = a.to(torch.int32)
        c = pattern()
        c = c.to(torch.int32)
        b = xor(a,c)
        b = b.to(torch.int32)

        x_prime = self.RAPP_Generator_model(x, b)
        z = self.RAPP_Facematcher_model(x_prime)

        z = standardize_tensor(z)

        logits = self.forward(z)

        loss = F.cross_entropy(logits, s.long())
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        train_acc, train_misclass = self.get_stats(logits, s)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u, a, s = batch

        a = a.to(torch.int32)
        c = pattern()
        c = c.to(torch.int32)
        b = xor(a, c)
        b = b.to(torch.int32)

        x_prime = self.RAPP_Generator_model(x, b)
        z = self.RAPP_Facematcher_model(x_prime)

        z = standardize_tensor(z)

        logits = self.forward(z)

        loss = F.cross_entropy(logits, s.long())
        self.log('val_loss', loss, on_step=True, prog_bar=True)

        acc, misclass = self.get_stats(logits, s)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, u, a, s = batch

        a = a.to(torch.int32)
        c = pattern()
        c = c.to(torch.int32)
        b = xor(a, c)
        b = b.to(torch.int32)

        x_prime = self.RAPP_Generator_model(x, b)
        z = self.RAPP_Facematcher_model(x_prime)

        z = standardize_tensor(z)

        logits = self.forward(z)

        loss = F.cross_entropy(logits, s.long())
        self.log('test_loss_' + self.dataset_name, loss, on_step=True, prog_bar=True)

        acc, train_misclass = self.get_stats(logits, s)
        self.log('test_acc_'+self.dataset_name, acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss', loss, 'test_s_accuracy', acc}

def GenderLogisticRegressionAttack(celeba_data_dir, pin_memory, fast_dev_run):
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/RAPP_LR_Gender/checkpoints')


    data_module_celeba_training = CelebaRAPPMineTrainingDatasetInterface(num_workers=0, dataset_name='CelebA', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=pin_memory)
    data_module_lfw = LFWRAPPMineDatasetInterface(num_workers=0, dataset_name='CelebA', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=pin_memory)
    data_module_adience = AdienceRAPPMineDatasetInterface(num_workers=0, dataset_name='CelebA', batch_size=256, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=pin_memory)
    logger_celeba_train = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='gender_LR_attack_logger_celeba')

    trainer_celeba_training = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode='min',
                monitor='train_loss',
                dirpath=os.path.join(CHECKPOINT_PATH, 'gender_LR_attack_models'),
                save_last=True,
                every_n_train_steps=50
            ),
            LearningRateMonitor('epoch'),
            EarlyStopping(
                monitor='val_acc',
                patience=5,
                mode='max'
            )
        ],

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'gender_LR_attack_models'),
        accelerator='auto',
        devices=1,
        max_epochs=100,
        min_epochs=50,
        logger=logger_celeba_train,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=fast_dev_run

    )

    print('RAPP models on Logistic Regression Gender ')
    model = RAPPLogisticRegressionAttack(dataset_name='CelebA_LFW_Adience')
    trainer_celeba_training.fit(model, data_module_celeba_training)
    trainer_celeba_training.test(model, data_module_celeba_training)
    trainer_celeba_training.test(model, data_module_lfw)
    trainer_celeba_training.test(model, data_module_adience)


def RaceLogisticRegressionAttack(celeba_data_dir, pin_memory, fast_dev_run):
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/RAPP_LR_Race/checkpoints')

    data_module_celeba_training = CelebaRAPPMineTrainingDatasetInterface(num_workers=0, dataset_name='CelebA',
                                                                         batch_size=256, dim_img=224,
                                                                         data_dir=celeba_data_dir, identity_nums=10177,
                                                                         sensitive_attr='Race', pin_memory=pin_memory)

    data_module_lfw = LFWRAPPMineDatasetInterface(num_workers=0, dataset_name='CelebA', batch_size=256, dim_img=224,
                                                  data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race',
                                                  pin_memory=pin_memory)
    data_module_adience = AdienceRAPPMineDatasetInterface(num_workers=0, dataset_name='CelebA', batch_size=256,
                                                          dim_img=224, data_dir=celeba_data_dir, identity_nums=10177,
                                                          sensitive_attr='Race', pin_memory=pin_memory)

    logger_celeba_train = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='race_LR_attack_logger_celeba')

    trainer_celeba_training = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode='min',
                monitor='train_loss',
                dirpath=os.path.join(CHECKPOINT_PATH, 'race_LR_attack_models'),
                save_last=True,
                every_n_train_steps=50
            ),
            LearningRateMonitor('epoch'),
            EarlyStopping(
                monitor='val_acc',
                patience=5,
                mode='max'
            )
        ],

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'race_LR_attack_models'),
        accelerator='auto',
        devices=1,
        max_epochs=100,
        min_epochs=50,
        logger=logger_celeba_train,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=fast_dev_run

    )

    print('RAPP models on Logistic Regression Race ')
    model = RAPPLogisticRegressionAttack(dataset_name='CelebA_LFW_Adience')
    trainer_celeba_training.fit(model, data_module_celeba_training)
    trainer_celeba_training.test(model, data_module_celeba_training)
    trainer_celeba_training.test(model, data_module_lfw)
    trainer_celeba_training.test(model, data_module_adience)







































