import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from arcface_resnet50 import ArcfaceResnet50
from model import BottleneckNets, Encoder, Decoder
import numpy as np
import os
import math
from pytorch_lightning.loggers import TensorBoardLogger
from data import CelebaInterface, LFWInterface, AdienceInterface, CelebaRaceInterface, CelebaAttackInterface, LFWCasiaInterface
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


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

class MultipleLayerInception(pl.LightningModule):
    def __init__(self, latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name):
        super(MultipleLayerInception, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.dataset_name = dataset_name

        self.pretrained_model_name = pretrained_model_name
        if pretrained_model_name == 'Arcface':
            arcface_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
            self.pretrained_model = arcface_net.load_from_checkpoint(r'E:\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
            self.pretrained_model.requires_grad_(False)


            #print(self.pretrained_model.resnet50)

        elif pretrained_model_name == 'Bottleneck':
            arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.5)
            arcface = arcface_resnet50_net.load_from_checkpoint(r'E:\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
            encoder = Encoder(latent_dim=latent_dim, arcface_model=arcface)
            decoder = Decoder(latent_dim=latent_dim, identity_nums=10177, s=64.0, m=0.5, easy_margin=False)
            bottlenecknets = BottleneckNets(model_name='bottleneck', encoder=encoder, decoder=decoder, beta=beta,
                                            batch_size=64, identity_nums=10177)
            bottlenecknets_pretrained_model = bottlenecknets.load_from_checkpoint(pretrained_model_path, encoder=encoder, decoder=decoder)
            self.pretrained_model = bottlenecknets_pretrained_model
            self.pretrained_model.requires_grad_(False)


    def forward(self, z):
        logits = self.mlp(z)
        return logits

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.mlp.parameters(), lr=0.001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode="min", factor=0.1, patience=5, min_lr=1e-8,
                                                         verbose=True, threshold=1e-3)
        return {"optimizer": optim_train, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def get_stats(self, logits, labels):
        preds =  torch.argmax(logits, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def training_step(self, batch, batch_idx):
        x, u, s = batch

        s = s.squeeze()

        if self.pretrained_model_name == 'Arcface':
            _, z = self.pretrained_model(x,u)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        elif self.pretrained_model_name == 'Bottleneck':
            z, _, _, _ = self.pretrained_model(x, u)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        logits = self.forward(z)
        loss = F.cross_entropy(logits, s.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        train_acc, train_misclass = self.get_stats(logits, s)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, u, s = batch
        s = s.squeeze()
        if self.pretrained_model_name == 'Arcface':
            _, z = self.pretrained_model(x, u)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        elif self.pretrained_model_name == 'Bottleneck':
            z, _, _, _ = self.pretrained_model(x, u)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        logits = self.forward(z)
        loss = F.cross_entropy(logits, s.long())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        val_acc, val_misclass = self.get_stats(logits, s)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_u_accuracy': val_acc}

    def test_step(self, batch, batch_idx):
        x, u, s = batch
        s = s.squeeze()

        if self.pretrained_model_name == 'Arcface':
            z = self.pretrained_model.resnet50(x)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        elif self.pretrained_model_name == 'Bottleneck':
            z, _, _, _ = self.pretrained_model(x, u)
            #z = F.normalize(z, p=2, dim=1)
            z = standardize_tensor(z)

        logits = self.forward(z)
        loss = F.cross_entropy(logits, s.long())
        self.log('test_loss'+self.dataset_name, loss, on_step=True, on_epoch=True, prog_bar=True)

        test_acc, val_misclass = self.get_stats(logits, s)
        self.log('test_acc' + self.dataset_name, test_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {'test_loss': loss, 'test_u_accuracy': test_acc}



def MLPGenderAttack(latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name):

    mlp_attack_model = MultipleLayerInception(latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name)

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/MLP_gender_attack/checkpoints/'+ pretrained_model_name + beta)

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='MLP_gender_logger'+str(dataset_name))  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    '''
    
    celeba_data_module = CelebaAttackInterface(
        num_workers=2,
        dataset='celeba_data',
        batch_size=256,
        dim_img=224,
        data_dir='D:\datasets\celeba',  # 'D:\datasets\celeba'
        sensitive_dim=1,
        identity_nums=10177,
        sensitive_attr='Male',
        pin_memory=False
    )

    lfw_data_module = LFWInterface(num_workers=2,
                               dataset='lfw',
                               data_dir='D:\datasets\lfw\lfw112',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='Male',
                               purpose='attr_extract',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)

    adience_data_module = AdienceInterface(num_workers=2,
                                   dataset='adience',
                                   data_dir='D:\datasets\Adience',
                                   batch_size=256,
                                   dim_img=224,
                                   sensitive_attr='Male',
                                   purpose='gender_extract',
                                   pin_memory=False,
                                   identity_nums=5749,
                                   sensitive_dim=1)
    '''

    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    lfw_casia_data = LFWCasiaInterface(dim_img=224,
                                       batch_size=256,
                                       dataset='lfw_casia_data',
                                       sensitive_attr='Male',
                                       lfw_data_dir=lfw_data_dir,
                                       casia_data_dir=casia_data_dir,
                                       purpose='attr_extract')

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="min",
                monitor="train_loss",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model'),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_acc", min_delta=0.001, patience=3, verbose=False, mode="max")
        ],  # Log learning rate every epoch
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model'),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=10,
        min_epochs=10,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=5,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    print('Model will be created')
    #trainer.fit(mlp_attack_model, celeba_data_module)
    #trainer.test(mlp_attack_model, celeba_data_module, ckpt_path=r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\experiments\lightning_logs\MLP_gender_attack\checkpoints\Bottleneck0.1\saved_model\last.ckpt')
    #trainer.test(mlp_attack_model, lfw_data_module, ckpt_path=r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\experiments\lightning_logs\MLP_gender_attack\checkpoints\Bottleneck1.0\saved_model\last.ckpt')
    trainer.test(mlp_attack_model, lfw_casia_data,  ckpt_path=r'E:\Bottleneck_Nets\experiments\lightning_logs\MLP_gender_attack\checkpoints\Bottleneck0.0001\saved_model\last.ckpt')

def MLPRaceAttack(latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name):
    mlp_attack_model = MultipleLayerInception(latent_dim, pretrained_model_name, pretrained_model_path, beta,
                                              dataset_name)

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT',
                                     'lightning_logs/MLP_race_attack/checkpoints/' + pretrained_model_name + beta)

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH,
                               name='MLP_race_logger'+str(dataset_name))  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs
    '''
    
    celeba_data_module = CelebaRaceInterface(
        num_workers=1,
        dataset='celeba_data',
        batch_size=256,
        dim_img=224,
        data_dir='D:\datasets\celeba',  # 'D:\datasets\celeba'
        sensitive_dim=1,
        identity_nums=10177,
        pin_memory=False)


    lfw_data_module = LFWInterface(num_workers=2,
                               dataset='lfw',
                               data_dir='D:\datasets\lfw\lfw112',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='White',
                               purpose='attr_extract',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)

    adience_data_module = AdienceInterface(num_workers=2,
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

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="min",
                monitor="train_loss",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model'),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_acc", min_delta=0.001, patience=3, verbose=False, mode="max")
        ],  # Log learning rate every epoch
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model'),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=10,
        min_epochs=10,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=5,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    print('Model will be created')
    #trainer.fit(mlp_attack_model, celeba_data_module)
    #trainer.test(mlp_attack_model, celeba_data_module)
    #trainer.test(mlp_attack_model, lfw_data_module, ckpt_path=r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\experiments\lightning_logs\MLP_race_attack\checkpoints\Bottleneck0.001\saved_model\last.ckpt')
    trainer.test(mlp_attack_model, lfw_casia_data, ckpt_path=r'E:\Bottleneck_Nets\experiments\lightning_logs\MLP_race_attack\checkpoints\Bottleneck1.0\saved_model\last.ckpt')

if __name__ == '__main__':
    latent_dim = 512
    #pretrained_model_name = 'Arcface'
    #pretrained_model_path = 'None'
    #beta = 'None'

    #MLPGenderAttack(latent_dim, pretrained_model_name, pretrained_model_path, beta, 'casia_lfw')


    #MLPRaceAttack(latent_dim, pretrained_model_name, pretrained_model_path, beta, 'lfw_casia')
    #LogisticRegressionRaceAttack(latent_dim, pretrained_model_name, pretrained_model_path, beta, 'celeba')


    pretrained_model_name = 'Bottleneck'
    beta_arr = [0.0001]
    for beta in beta_arr:
        pretrained_model_path = r'E:\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta' + str(beta) + '\checkpoints\saved_models\last.ckpt'

        MLPGenderAttack(latent_dim, 'Bottleneck', pretrained_model_path, str(beta), 'lfw_casia')


    #beta_arr = [1.0]
    #for beta in beta_arr:
    #    pretrained_model_path = r'E:\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta' + str(beta) + '\checkpoints\saved_models\last.ckpt'
    #    MLPRaceAttack(latent_dim, 'Bottleneck', pretrained_model_path, str(beta), 'lfw_casia')

    #beta_arr = [0.0001, 0.001, 0.01, 0.1, 1.0]
    #for beta in beta_arr:
    #    pretrained_model_path = r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta' + str(
    #        beta) + '\checkpoints\saved_models\last.ckpt'

    #    LogisticRegressionRaceAttack(latent_dim, 'Bottleneck', pretrained_model_path, str(beta), 'celeba')

    #beta_arr = [0.1, 1.0]
    #for beta in beta_arr:
    #    pretrained_model_path = r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta' + str(
    #        beta) + '\checkpoints\saved_models\last.ckpt'

    #    Attack(latent_dim, 'Bottleneck', pretrained_model_path, str(beta), 'celeba')
