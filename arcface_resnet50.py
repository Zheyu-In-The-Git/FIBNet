import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.optim as optim
import os
import math
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from model import ResNet50, ArcMarginProduct

pl.seed_everything(83)


def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)



class ArcfaceResnet50(pl.LightningModule):
    def __init__(self):
        super(ArcfaceResnet50, self).__init__()
        self.resnet50 = ResNet50(512, channels=3)
        self.arc_margin_product = ArcMarginProduct(in_features=512, out_features=10177, s=30.0, m=0.50, easy_margin=False)
        self.softmax = nn.Softmax()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, u):
        z = self.resnet50(x)
        output = self.arc_margin_product(z, u)
        return output

    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.parameters(), lr=0.001, betas=(b1, b2))
        return optim_train

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def cosin_metric(self, x1, x2):
        return torch.tensor(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))


    def training_step(self, batch, batch_idx):
        x, u, _ = batch
        output = self(x, u)
        loss = self.criterion(output, u)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        acc, miss_rate = self.get_stats(output, u)
        self.log('train_acc', acc, on_step = True, on_epoch=True)
        self.log('train_miss_rate', miss_rate, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img_x, img_y, match = batch


        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        img_x, img_y, match = batch


        acc = (u == preds).float().mean()
        self.log("test_acc", acc)




CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/face_recognizer_resnet50_latent1024/checkpoints/')
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
                mode="max",
                monitor="val_acc",
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name)
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=120,
        min_epochs=100,
        logger=logger,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        check_val_every_n_epoch=10,
        fast_dev_run=False
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if Resume:
        # Automatically loads the model with the saved hyperparameters
        resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH,  'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        print('Found pretrained model at ' + resume_checkpoint_path + ', loading ... ')  # 重新加载
        model = FaceRecognizer()
        trainer.fit(model, data_module, ckpt_path=resume_checkpoint_path)

    else:
        resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        print('Model will be created')
        model = FaceRecognizer()
        trainer.fit(model, data_module)
        trainer.save_checkpoint(resume_checkpoint_path)




if __name__ == '__main__':
    main(model_name='face_recognition_resnet50',  Resume = 0, save_name=None)





