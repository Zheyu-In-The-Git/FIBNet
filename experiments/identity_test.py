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
from data import CelebaInterface, LFWInterface, AdienceInterface, CelebaRaceInterface, CelebaAttackInterface
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from logistic_regression_attack import LogisticRegression, Attack, LogisticRegressionRaceAttack
import torchmetrics

def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class IdentityTest(pl.LightningModule):
    def __init__(self, latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name):
        super(IdentityTest, self).__init__()
        self.dataset_name = dataset_name
        self.beta = beta

        self.pretrained_model_name = pretrained_model_name
        if pretrained_model_name == 'Arcface':
            arcface_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
            self.pretrained_model = arcface_net.load_from_checkpoint(
                r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')

            self.pretrained_model = self.pretrained_model.resnet50
            self.pretrained_model.requires_grad_(False)
            #print(self.pretrained_model)

        elif pretrained_model_name == 'Bottleneck':
            arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.5)
            arcface = arcface_resnet50_net.load_from_checkpoint(
                r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
            encoder = Encoder(latent_dim=latent_dim, arcface_model=arcface)
            decoder = Decoder(latent_dim=latent_dim, identity_nums=10177, s=64.0, m=0.5, easy_margin=False)
            bottlenecknets = BottleneckNets(model_name='bottleneck', encoder=encoder, decoder=decoder, beta=beta,
                                            batch_size=64, identity_nums=10177)
            bottlenecknets_pretrained_model = bottlenecknets.load_from_checkpoint(pretrained_model_path,
                                                                                  encoder=encoder, decoder=decoder)
            self.pretrained_model = bottlenecknets_pretrained_model.encoder
            self.pretrained_model.requires_grad_(False)

        # roc
        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, x):
        if self.pretrained_model_name == 'Arcface':
            z = self.pretrained_model(x)
            return z

        elif self.pretrained_model_name == 'Bottleneck':
            mu, log_var = self.pretrained_model(x)

            return mu, log_var

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

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 确实变成了标准差
        std = torch.clamp(std, min=1e-4)
        jitter = 1e-4
        eps = torch.randn_like(std)
        z = mu + eps * (std + jitter)
        return z


    def test_step(self, batch, batch_idx):
        # 数据
        img_1, img_2, match = batch

        z_1_mu, z_1_sigma = self.forward(img_1)
        z_2_mu, z_2_sigma = self.forward(img_2)

        z_1 = self.sample_z(z_1_mu, z_1_sigma)
        z_2 = self.sample_z(z_2_mu, z_2_sigma)

        return {'z_1':z_1, 'z_2':z_2, 'match':match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)
        self.log('eer_cos', eer_cos, on_epoch=True)
        bottleneck_net_confusion_cos = {'fpr_cos': fpr_cos, 'tpr_cos': tpr_cos, 'thresholds_cos': thresholds_cos,'eer_cos': eer_cos}

        torch.save(bottleneck_net_confusion_cos, r'lightning_logs/bottleneck_roc_beta'+str(self.beta)+str(self.dataset_name)+'.pt')



    '''
    
    def test_step(self, batch, batch_idx):
        img_1, img_2, match = batch
        z_1 = self.forward(img_1)
        z_2 = self.forward(img_2)
        return {'z_1': z_1, 'z_2': z_2, 'match': match}

    def test_epoch_end(self, outputs):
        match = torch.cat([x['match'] for x in outputs], dim=0)
        z_1 = torch.cat([x['z_1'] for x in outputs], dim=0)
        z_2 = torch.cat([x['z_2'] for x in outputs], dim=0)
        cos = F.cosine_similarity(z_1, z_2, dim=1)
        match = match.long()

        fpr_cos, tpr_cos, thresholds_cos, eer_cos = self.calculate_eer(cos, match)

        self.log('eer_cos', eer_cos, prog_bar=True)

        arcface_confusion_cos = {'fpr_cos':fpr_cos,'tpr_cos':tpr_cos,'thresholds_cos':thresholds_cos,'eer_cos':eer_cos}
    '''
        #torch.save(arcface_confusion_cos, r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\experiments\lightning_logs\roc_arcface_'+str(self.dataset_name)+'.pt')




def IdentityTestFunction(latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name):
    identity_test_model = IdentityTest(latent_dim, pretrained_model_name, pretrained_model_path, beta, dataset_name)
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/identity_test/bottleneck_checkpoints_Adience/' + pretrained_model_name + beta)
    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='identity_test')  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs
    lfw_data_module = LFWInterface(num_workers=2,
                               dataset='lfw',
                               data_dir='D:\datasets\lfw\lfw112',
                               batch_size=256,
                               dim_img=224,
                               sensitive_attr='Male',
                               purpose='face_recognition',
                               pin_memory=False,
                               identity_nums=5749,
                               sensitive_dim=1)

    adience_data_module = AdienceInterface(num_workers=2,
                                           dataset='adience',
                                           data_dir='D:\datasets\Adience',
                                           batch_size=256,
                                           dim_img=224,
                                           sensitive_attr='Male',
                                           purpose='face_recognition',
                                           pin_memory=False,
                                           identity_nums=5749,
                                           sensitive_dim=1)

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
    trainer.test(identity_test_model, adience_data_module)
    # trainer.test(logistic_attack_model, lfw_data_module)
    # trainer.test(logistic_attack_model, adience_data_module)

if __name__ == '__main__':
    latent_dim = 512

    #pretrained_model_name = 'Arcface'
    #pretrained_model_path = 'None'
    #beta = 'None'

    #IdentityTestFunction(latent_dim, pretrained_model_name, pretrained_model_path, beta, 'lfw')

    beta_arr = [0.0001, 0.001, 0.01, 0.1, 1.0]
    for beta in beta_arr:
        pretrained_model_path = r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta' + str(
            beta) + '\checkpoints\saved_models\last.ckpt'

        IdentityTestFunction(latent_dim, 'Bottleneck', pretrained_model_path, str(beta), 'Adience')



