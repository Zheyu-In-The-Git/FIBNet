import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.optim as optim
import os
import math
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from model import ResNet50, ArcMarginProduct
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.functional import mean_squared_error

pl.seed_everything(83)
import torch.nn.functional as F


class ArcfaceResnet50(pl.LightningModule):
    def __init__(self, in_features=512, out_features=10177, s=30.0, m=0.50):
        super(ArcfaceResnet50, self).__init__()
        self.resnet50 = ResNet50(512, channels=3)
        self.arc_margin_product = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m, easy_margin=False)
        self.softmax = nn.Softmax()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

        # 使用一些度量
        # 欧几里得距离
        self.pdist = nn.PairwiseDistance(p=2)

        # 预测准确度
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10177)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10177)
        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, x, u):
        z = self.resnet50(x)
        output = self.arc_margin_product(z, u)
        return output, z

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.parameters(), lr=0.001, betas=(b1, b2))
        return optim_train

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

    def training_step(self, batch, batch_idx):
        x, u, _ = batch
        output, _ = self.forward(x, u)
        loss = self.criterion(output, u)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.train_acc(output, u)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u, _ = batch
        output, _ = self.forward(x, u)
        self.valid_acc(output, u)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        loss = self.criterion(output, u)
        self.log('valid_loss', loss, on_step=True, on_epoch=True)

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
        dist = self.pdist(z_1, z_2)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)
        fpr_dist, tpr_dist, thresholds_dist, eer_dist = self.calculate_eer(dist, match)

        arcface_confusion_cos = {'fpr_cos':fpr_cos,'tpr_cos':tpr_cos,'thresholds_coss':thresholds_cos,'eer_cos':eer_cos}
        torch.save( arcface_confusion_cos, r"C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_confusion_cos.pt")

        arcface_confusion_dist = {'fpr_dist':fpr_dist,'tpr_mse':tpr_dist,'thresholds_dist':thresholds_dist,'eer_mse':eer_dist}
        torch.save(arcface_confusion_dist, r"C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_confusion_dist.pt")



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

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH + '/lightning_log', name='tensorboard_log', version='face_recognizer_resnet50_logger' )  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="min",
                monitor="valid_loss_epoch",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name)
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
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
        check_val_every_n_epoch=10,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if Resume:
        # Automatically loads the model with the saved hyperparameters
        #resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH,  'saved_models')
        #os.makedirs(resume_checkpoint_dir, exist_ok=True)
        #resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        #print('Found pretrained model at ' + resume_checkpoint_path + ', loading ... ')  # 重新加载
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=30.0, m=0.50)
        trainer.fit(model, data_module, ckpt_path='lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=19-step=39900.ckpt')
        trainer.test(model, data_module)

    else:
        resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        print('Model will be created')
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=30.0, m=0.50)
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        trainer.save_checkpoint(resume_checkpoint_path)




if __name__ == '__main__':
    main(model_name='face_recognition_resnet50',  Resume = 1, save_name=None)





