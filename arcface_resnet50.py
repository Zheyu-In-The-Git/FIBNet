import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np
import math
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from model import ResNet50, ArcMarginProduct, FocalLoss
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.functional import mean_squared_error


pl.seed_everything(83)
import torch.nn.functional as F


def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class ArcfaceResnet50(pl.LightningModule):
    def __init__(self, in_features=512, out_features=10177, s=30.0, m=0.50):
        super(ArcfaceResnet50, self).__init__()
        self.resnet50 = ResNet50(512, channels=3)
        self.arc_margin_product = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m, easy_margin=False)
        self.softmax = nn.Softmax()
        self.save_hyperparameters()
        self.criterion = FocalLoss(gamma=2)

        # 使用一些度量
        # 欧几里得距离
        self.pdist = nn.PairwiseDistance(p=2)

        # roc
        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, x, u):
        z = self.resnet50(x)
        output = self.arc_margin_product(z, u)
        return output, z

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.parameters(), lr=0.0005, betas=(b1, b2), weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_train, mode="min", factor=0.1, patience=3, min_lr=1e-8, threshold=1e-2)
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

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def training_step(self, batch, batch_idx):
        x, u, _ = batch
        output, _ = self.forward(x, u)
        loss = self.criterion(output, u)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        train_acc, train_misclass = self.get_stats(output, u)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_misclass', train_misclass, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u, _ = batch
        output, _ = self.forward(x, u)
        loss = self.criterion(output, u)
        self.log('valid_loss', loss, on_step=True, on_epoch=True)

        valid_acc, valid_misclass = self.get_stats(output, u)
        self.log('valid_acc', valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('valid_misclass', valid_misclass, on_step=True, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        img_1, img_2, match = batch
        z_1 = self.resnet50(img_1)
        z_2 = self.resnet50(img_2)
        return {'z_1':z_1, 'z_2':z_2, 'match':match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)

        self.log('eer_cos', eer_cos, on_step=True, on_epoch=True, prog_bar=True)

        arcface_confusion_cos = {'fpr_cos':fpr_cos,'tpr_cos':tpr_cos,'thresholds_cos':thresholds_cos,'eer_cos':eer_cos}
        torch.save(arcface_confusion_cos,r"/lightning_logs")



CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/')
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def main(model_name, Resume, save_name=None):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    data_module = CelebaInterface(num_workers =2,
                 dataset = 'celeba_data',
                 batch_size = 64,
                 dim_img = 224,
                 data_dir = 'D:\datasets\celeba', # 'D:\datasets\celeba'
                 sensitive_dim = 1,
                 identity_nums = 10177,
                 sensitive_attr = 'Male',
                 pin_memory=False)

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH + '/lightning_log', name='tensorboard_log')  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="max",#最大
                monitor="valid_acc_epoch",#用valid_acc比较好
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor='valid_acc_epoch',#用valid_acc比较好
                patience = 5,#用valid_acc比较好5吧 间隔为3个epoch为一周期
                mode='max'#用valid_acc比较好 保存top_k 3个比较好把
            )
        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=200,
        min_epochs=100,
        logger=logger,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        check_val_every_n_epoch=10,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1,
        auto_lr_find=True,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if Resume:
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=32.0, m=0.50)
        trainer.fit(model, data_module, ckpt_path='lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/last.ckpt')
        trainer.test(model, data_module)
        trainer.save_checkpoint('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50')
    else:
        resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        print('Model will be created')
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=32.0, m=0.50)
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        trainer.save_checkpoint(resume_checkpoint_path)




if __name__ == '__main__':
    main(model_name='face_recognition_resnet50',  Resume = 0, save_name=None)





