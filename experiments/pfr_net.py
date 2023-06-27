import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from arcface_resnet50 import ArcfaceResnet50
from model import BottleneckNets, Encoder, Decoder
import numpy as np
import os
from pytorch_lightning.loggers import TensorBoardLogger
from data import CelebaInterface, LFWInterface, AdienceInterface, CelebaRaceInterface, CelebaAttackInterface
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import math
import torchmetrics


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

class PFRNet(pl.LightningModule):
    def __init__(self, latent_dim):
        super(PFRNet, self).__init__()

        self.Encoder_ind = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 496)
        )
        self.Encoder_dep = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.Decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

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

        arcface_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
        self.pretrained_model = arcface_net.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\epoch=48-step=95800.ckpt')
        self.pretrained_model.requires_grad_(False)

        self.MSE = nn.MSELoss()

        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, z):
        z_ind = self.Encoder_ind(z)
        z_dep = self.Encoder_dep(z)
        z = torch.cat((z_ind, z_dep), dim=1)
        z_new = self.Decoder(z)
        return z_new, z_ind, z_dep

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.parameters(), lr=0.0001, betas=(b1, b2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode="min", factor=0.1, patience=5, min_lr=1e-8,
                                                         verbose=True, threshold=1e-3)
        return {"optimizer": optim_train, "lr_scheduler": scheduler, "monitor": "train_loss"}

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


    def get_stats(self, logits, labels):
        preds =  torch.argmax(logits, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def alpha_order_moment(self, x, alpha):
        return torch.mean(torch.pow(x, alpha), dim=0)

    def training_step(self, batch, batch_idx):
        x, u, s = batch

        s = s.squeeze()

        _, z = self.pretrained_model(x, u)

        z_new, z_ind, z_dep = self.forward(z)

        female_mask = (s == 0)
        female_z_ind = z_ind[female_mask]

        male_mask = (s == 1)
        male_z_ind = z_ind[male_mask]

        alpha_1_order_moment_f = self.alpha_order_moment(female_z_ind,1)
        alpha_1_order_moment_m = self.alpha_order_moment(male_z_ind,1)

        alpha_2_order_moment_f = self.alpha_order_moment(female_z_ind,2)
        alpha_2_order_moment_m = self.alpha_order_moment(female_z_ind,2)

        l_1 = self.MSE(alpha_1_order_moment_f, alpha_1_order_moment_m) + self.MSE(alpha_2_order_moment_f, alpha_2_order_moment_m)

        female_mask = (s == 0)
        female_z_dep = z_dep[female_mask]

        male_mask = (s == 1)
        male_z_dep = z_dep[male_mask]

        beta_1_order_moment_f = self.alpha_order_moment(female_z_dep,1)
        beta_1_order_moment_m = self.alpha_order_moment(male_z_dep, 1)

        beta_2_order_moment_f = self.alpha_order_moment(female_z_dep, 2)
        beta_2_order_moment_m = self.alpha_order_moment(male_z_dep, 2)


        l_2 = torch.mean(torch.exp(- torch.square(torch.abs(beta_1_order_moment_f - beta_1_order_moment_m))) + torch.exp(- torch.square(torch.abs(beta_2_order_moment_f - beta_2_order_moment_m))))

        train_loss = self.MSE(z_new, z) + 0.01 * l_1 + 0.01 * l_2

        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, u, s = batch
        s = s.squeeze()

        _, z = self.pretrained_model(x, u)

        z_new, z_ind, z_dep = self.forward(z)

        female_mask = (s == 0)
        female_z_ind = z_ind[female_mask]

        male_mask = (s == 1)
        male_z_ind = z_ind[male_mask]

        alpha_1_order_moment_f = self.alpha_order_moment(female_z_ind,1)
        alpha_1_order_moment_m = self.alpha_order_moment(male_z_ind,1)

        alpha_2_order_moment_f = self.alpha_order_moment(female_z_ind,2)
        alpha_2_order_moment_m = self.alpha_order_moment(female_z_ind,2)

        l_1 = self.MSE(alpha_1_order_moment_f, alpha_1_order_moment_m) + self.MSE(alpha_2_order_moment_f, alpha_2_order_moment_m)

        female_mask = (s == 0)
        female_z_dep = z_dep[female_mask]

        male_mask = (s == 1)
        male_z_dep = z_dep[male_mask]

        beta_1_order_moment_f = self.alpha_order_moment(female_z_dep,1)
        beta_1_order_moment_m = self.alpha_order_moment(male_z_dep, 1)

        beta_2_order_moment_f = self.alpha_order_moment(female_z_dep, 2)
        beta_2_order_moment_m = self.alpha_order_moment(male_z_dep, 2)


        l_2 = torch.mean(torch.exp(- torch.square(torch.abs(beta_1_order_moment_f - beta_1_order_moment_m))) + torch.exp(- torch.square(torch.abs(beta_2_order_moment_f - beta_2_order_moment_m))))


        valid_loss = self.MSE(z_new, z) + 0.01 * l_1 + 0.01 * l_2

        self.log('valid_loss', valid_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return valid_loss

    def test_step(self, batch, batch_idx):
        img_1, img_2, match = batch

        arcface_model_resnet50 = self.pretrained_model.resnet50

        z_1 = arcface_model_resnet50(img_1)
        z_2 = arcface_model_resnet50(img_2)

        z_1, _, _ = self.forward(z_1)
        z_2, _, _ = self.forward(z_2)
        return {'z_1': z_1, 'z_2': z_2, 'match': match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)

        self.log('eer_cos', eer_cos, prog_bar=True)

        PFRNet_confusion_cos = {'fpr_cos':fpr_cos,'tpr_cos':tpr_cos,'thresholds_cos':thresholds_cos,'eer_cos':eer_cos}
        torch.save(PFRNet_confusion_cos, r'lightning_logs/PFRNet_confusion_cos.pt')





def PFRNetExperiment():
    PFRNet_model = PFRNet(512)

    data_module = CelebaInterface(num_workers=2,
                                  dataset='celeba_data',
                                  batch_size=100,
                                  dim_img=224,
                                  data_dir='D:\datasets\celeba',  # 'D:\datasets\celeba'
                                  sensitive_dim=1,
                                  identity_nums=10177,
                                  sensitive_attr='Male',
                                  pin_memory=False)

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/PFRNet_gender/checkpoints_celebatest/')
    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='PFRNet_gender_logger')

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="min",
                monitor="loss",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model'),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),

        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model'),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=130,
        min_epochs=100,
        logger=logger,
        log_every_n_steps=50,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    print('Model will be created')
    trainer.fit(PFRNet_model, data_module)
    trainer.test(PFRNet_model, data_module)

if __name__ == '__main__':
    PFRNetExperiment()

















