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
import torchmetrics

from RAPP import RAPP, Generator, Discriminator
from RAPP_data_interface import CelebaRAPPDatasetInterface



device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #******#
# password 全为1的向量 []
# seed 全为0101的向量
def xor(a, b):
    a = a.to(device)
    b = b.to(device)
    c = torch.logical_xor(a, b).int()
    c = c.to(device)
    return c


def pattern():
    vector_length = 10
    pattern = torch.tensor([1, 0, 1, 0])
    vector = torch.cat([pattern[i % 4].unsqueeze(0) for i in range(vector_length)])
    vector = vector.to(torch.int32)
    return vector




class RAPPIdentityTest(pl.LightningModule):
    def __init__(self, dataset_name):
        super(RAPPIdentityTest, self).__init__()
        self.dataset_name = dataset_name

        # 创建RAPP 网络
        RAPP_model = RAPP()
        RAPP_model = RAPP_model.load_from_checkpoint(os.path.abspath(r'lightning_logs/RAPP_checkpoints/saved_model/last.ckpt'))
        self.RAPP_model = RAPP_model
        self.RAPP_model.requires_grad_(False)

        # 生成器
        self.generator = self.RAPP_model.generator
        self.generator.requires_grad_(False)

        arcface_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
        pretrained_model = arcface_net.load_from_checkpoint(r'E:\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')

        self.face_match = pretrained_model.resnet50
        self.face_match.requires_grad_(False)

        # roc
        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, x):
        return x

    def calculate_eer(self, metrics, match):
        fpr, tpr, thresholds = self.roc(metrics, match)
        eer = 1.0
        min_dist = 1.0
        for i in range(len(fpr)):
            dist = abs(fpr[i] - (1 - tpr[i]))
            if dist < min_dist:
                min_dist = dist
                eer = (fpr[i] + (1 - tpr[i])) / 2
        return fpr, tpr, thresholds, eer

    def test_step(self, batch, batch_idx):
        img_1, img_2, match= batch

        batch_size = img_1.size(0)

        a = pattern()
        c = pattern()
        b = xor(a, c)
        #print(b.size())
        a = a.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        a = a.expand(batch_size, -1, -1, -1)

        b = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        b = b.expand(batch_size, -1, -1, -1)
        #print(b.size())

        a = a.to(device)
        b = b.to(device)

        img_1 = self.generator(self.generator(img_1, b), a)
        img_2 = self.generator(self.generator(img_2, b), a)

        z_1 = self.face_match(img_1)
        z_2 = self.face_match(img_2)
        return {'z_1': z_1, 'z_2': z_2, 'match': match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)

        self.log('eer_cos', eer_cos, prog_bar=True)

        RAPP_confusion_cos = {'fpr_cos': fpr_cos, 'tpr_cos': tpr_cos, 'thresholds_cos': thresholds_cos,
                              'eer_cos': eer_cos}
        torch.save(RAPP_confusion_cos, 'lightning_logs/'+self.dataset_name+'_RAPP_cos.pt')


def IdentityTestFunction(dataset_name):
    identity_test_model = RAPPIdentityTest(dataset_name)
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/identity_test/'+dataset_name)
    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='identity_test'+dataset_name)  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs
    celeba_data_module = CelebaRAPPDatasetInterface(num_workers=3,
                                                    dataset='celeba_data',
                                                    batch_size=64,
                                                    dim_img=224,
                                                    data_dir='E:\datasets\celeba',  # 'D:\datasets\celeba'
                                                    sensitive_dim=1,
                                                    identity_nums=10177,
                                                    pin_memory=False)
    lfw_data_module = LFWInterface(num_workers=2,
                               dataset='lfw',
                               data_dir='E:\datasets\lfw\lfw112',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='Male',
                               purpose='face_recognition',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)

    adience_data_module = AdienceInterface(num_workers=2,
                                           dataset='adience',
                                           data_dir='E:\datasets\Adience',
                                           batch_size=256,
                                           dim_img=224,
                                           sensitive_attr='Male',
                                           purpose='face_recognition',
                                           pin_memory=False,
                                           identity_nums=5749,
                                           sensitive_dim=1)

    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    lfw_casia_dataloader = LFWCasiaInterface(dim_img=224,
                                        batch_size=100,
                                        dataset='lfw_casia_data',
                                        sensitive_attr='White',
                                        lfw_data_dir=lfw_data_dir,
                                        casia_data_dir=casia_data_dir,
                                        purpose='face_recognition')

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model'),  # Where to save models
        accelerator="auto",
        devices=1,
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
    #trainer.fit(identity_test_model, celeba_data_module)
    if dataset_name == 'CelebA':
        trainer.test(identity_test_model, celeba_data_module)
    elif dataset_name == 'LFW':
        trainer.test(identity_test_model, lfw_data_module)
    elif dataset_name == 'LFW_CASIA':
        trainer.test(identity_test_model, lfw_casia_dataloader)
    else:
        trainer.test(identity_test_model, adience_data_module)



if __name__ == '__main__':
    #IdentityTestFunction('CelebA')
    #IdentityTestFunction('LFW')
    #IdentityTestFunction('Adience')
    IdentityTestFunction('LFW_CASIA')

