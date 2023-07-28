import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torchvision.utils
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from model import ResNet50, ArcMarginProduct, FocalLoss
import torchvision.models as models
import pickle
from inception_resnet_v1 import InceptionResnetV1
import itertools
from torchvision import transforms
import math

pl.seed_everything(83)
import torch.nn.functional as F
from RAPP_data_interface import CelebaRAPPDatasetInterface

from data.lfw_interface import LFWInterface
from data.adience_interface import AdienceInterface




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
    pattern = torch.tensor([0, 1, 0, 1])
    vector = torch.cat([pattern[i % 4].unsqueeze(0) for i in range(vector_length)])
    vector = vector.to(torch.int32)
    return vector




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        Conv_BN_IN_LReLU = []
        Conv_BN_IN_LReLU.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        #Conv_BN_IN_LReLU.append(nn.BatchNorm2d(out_channels))
        Conv_BN_IN_LReLU.append(nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))
        Conv_BN_IN_LReLU.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))

        Conv_BN_IN_LReLU.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        Conv_BN_IN_LReLU.append(nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))
        Conv_BN_IN_LReLU.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))

        self.conv_bn_in_leakyrelu = nn.Sequential(*Conv_BN_IN_LReLU)

    def forward(self, x):
        out = self.conv_bn_in_leakyrelu(x)
        return out

class ConvTransporseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransporseBlock, self).__init__()
        ConvTransporse_BN_IN_LReLU = []
        ConvTransporse_BN_IN_LReLU.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        #ConvTransporse_BN_IN_LReLU.append(nn.BatchNorm2d(out_channels))
        ConvTransporse_BN_IN_LReLU.append(nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))
        ConvTransporse_BN_IN_LReLU.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))

        #ConvTransporse_BN_IN_LReLU.append(nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=1, padding=1))
        #ConvTransporse_BN_IN_LReLU.append(nn.BatchNorm2d(out_channels))
        #ConvTransporse_BN_IN_LReLU.append(nn.LeakyReLU())


        self.convtransporse_bn_in_leakyrelu = nn.Sequential(*ConvTransporse_BN_IN_LReLU)

    def forward(self, x):
        out = self.convtransporse_bn_in_leakyrelu(x)
        return out



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.left_conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.left_conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.left_conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.left_conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.left_conv5 = ConvBlock(in_channels=512, out_channels=1024)

        self.right_conv5 = ConvTransporseBlock(in_channels=1024+10, out_channels=1024-512) # 1024+40
        self.right_conv4 = ConvTransporseBlock(in_channels=1024, out_channels=512 - 256)
        self.right_conv3 = ConvTransporseBlock(in_channels=512, out_channels=256)
        self.right_conv2 = ConvTransporseBlock(in_channels=256, out_channels=128)

        self.right_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x, s):
        x = self.left_conv1(x)
        x = self.left_conv2(x)
        x = self.left_conv3(x)
        x3_identity = x
        x = self.left_conv4(x)
        x4_identity = x
        x = self.left_conv5(x)

        s = s.view(s.size(0), s.size(1), 1, 1)
        s = s.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x,s], dim=1)

        x = self.right_conv5(x)
        x = torch.cat((x, x4_identity), dim=1)
        x = self.right_conv4(x)
        x = torch.cat((x, x3_identity), dim=1)
        x = self.right_conv3(x)
        x = self.right_conv2(x)
        x = self.right_conv1(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv5 = ConvBlock(in_channels=512, out_channels=1024)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1_1 = nn.Linear(16, 1)
        self.fc1_2 = nn.Linear(1024,1)
        self.fc2_1 = nn.Linear(16, 1)
        self.fc2_2 = nn.Linear(1024, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.adaptive_avgpool(x)
        x = x.view(x.size(0), x.size(1), 4*4)


        real = self.fc1_1(x)
        real = real.view(real.size(0), real.size(2),  real.size(1)) # 不行的话这里改一下
        real = self.fc1_2(real)
        real = real.view(real.size(0), real.size(1) * real.size(2))
        #real = self.sigmoid(real) # 到时候用 torch.nn.BCELoss()看下可不可以

        sensitive_attribute = self.fc2_1(x)
        sensitive_attribute = sensitive_attribute.view(sensitive_attribute.size(0), sensitive_attribute.size(2), sensitive_attribute.size(1))
        sensitive_attribute = self.fc2_2(sensitive_attribute)
        sensitive_attribute = sensitive_attribute.view(sensitive_attribute.size(0), sensitive_attribute.size(2)*sensitive_attribute.size(1))


        return real, sensitive_attribute


class FaceMatch(nn.Module):
    def __init__(self):
        super(FaceMatch, self).__init__()
        face_match_net = InceptionResnetV1(pretrained='vggface2')
        facenet_vggface2_path = r'E:\Bottleneck_Nets\RAPP_experiments\lightning_logs\facenet-vggface2.pt'
        face_match_net.load_state_dict(torch.load(facenet_vggface2_path))
        face_match_net.last_bn = nn.Sequential()

        self.face_match_model = face_match_net

    def forward(self, x):
        return self.face_match_model(x)



class RAPP(pl.LightningModule):
    def __init__(self):
        super(RAPP, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()




        self.face_match = FaceMatch()


        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.unnormalize_img = transforms.Normalize(mean=[-m/s for m,s in zip(mean, std)], std=[1/s for s in std])


        self.bcewithlogits = nn.BCEWithLogitsLoss()
        self.l1_norm = nn.L1Loss()

        self.roc = torchmetrics.ROC(task='binary')


    def forward(self, x, s):
        x_prime = self.generator(x, s)
        real, sensitive_attribute = self.discriminator(x_prime)
        face_representation = self.face_match(x)

        return face_representation, real, sensitive_attribute

    def gradient_penalty(self, x, x_prime):

        batch_size = x.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated_imgs = alpha * x + (1-alpha) * x_prime
        interpolated_imgs.requires_grad_(True)

        score, _ = self.discriminator(interpolated_imgs)


        gradients = torch.autograd.grad(outputs=score, inputs=interpolated_imgs, grad_outputs=torch.ones(score.size()).to(device), create_graph=True, retain_graph=True)

        gradient_norms = [(grad.norm(2, dim=1) - 1) ** 2 for grad in gradients]
        gradient_penalty = torch.stack(gradient_norms).mean()
        return gradient_penalty


    def configure_optimizers(self):
        n_critic = 5

        lr = 0.0002
        beta1 = 0.5
        beta2 = 0.99

        opt_g_fm = optim.Adam(itertools.chain(self.generator.parameters(), self.face_match.parameters()), lr=lr, betas=(beta1, beta2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        lr_g_fm = optim.lr_scheduler.ReduceLROnPlateau(opt_g_fm, mode="min", factor=0.5, patience=2, min_lr=1e-7,
                                                         verbose=True, threshold=1e-2)
        lr_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.5, patience=2, min_lr=1e-7,
                                                         verbose=True, threshold=1e-2)

        return ({'optimizer': opt_g_fm, 'frequency':1, "lr_scheduler": {'scheduler':lr_g_fm, "monitor": "loss_total_G"}},
                {'optimizer': opt_d, 'frequency': n_critic, "lr_scheduler": {'scheduler':lr_d, "monitor": "loss_total_D_C"}})

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, u, attribute = batch

        a = attribute
        c = pattern()
        b = xor(a, c)

        a = a.to(torch.int32)
        c = c.to(torch.int32)
        b = b.to(torch.int32)

        # 生成器的超参数
        lambda_attr_g = 10
        lambda_rec = 20
        lambda_m = 10

        # 判别器的超参数
        lambda_gp = 10
        lambda_attr_c = 1

        if optimizer_idx == 0:


            if self.global_step % 50 == 0:
                # 挑5张原图，并且反归一化
                original_imgs = x[:5]
                original_imgs_rgb = original_imgs.clone()
                original_imgs_rgb = self.unnormalize_img(original_imgs_rgb)
                grid = torchvision.utils.make_grid(original_imgs_rgb)
                self.logger.experiment.add_image('original_imgs', grid, self.global_step)

            x_prime = self.generator(x, b)

            if self.global_step % 50 == 0:
                # 挑5张生成，并且反归一化
                generated_imgs = x_prime[:5]
                generated_imgs_rgb = generated_imgs.clone()
                generated_imgs_rgb = self.unnormalize_img(generated_imgs_rgb)
                grid = torchvision.utils.make_grid(generated_imgs_rgb)
                self.logger.experiment.add_image('generated_imgs', grid, self.global_step)

            discriminator_output_fake, sensitive_attribute = self.discriminator(x_prime)
            loss_adv_G = - torch.mean(discriminator_output_fake)

            # 开始写 loss_attr_G
            loss_attr_G = self.bcewithlogits(sensitive_attribute, b.float())

            # identity loss
            loss_m_G = torch.mean(1 - F.cosine_similarity(self.face_match(x), self.face_match(x_prime), dim=1))

            # reconstruction loss
            loss_rec_G = self.l1_norm(self.generator(x_prime, a), x)

            loss_total_G = loss_adv_G + lambda_attr_g * loss_attr_G + lambda_rec * loss_rec_G + lambda_m * loss_m_G


            self.log('loss_adv_G', loss_adv_G, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_attr_G ', loss_attr_G, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_rec_G', loss_rec_G, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_m_G', loss_m_G, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_total_G', loss_total_G, on_step=True, on_epoch=True, prog_bar=True)

            return loss_total_G


        if optimizer_idx == 1: # 训练判别器

            x_prime = self.generator(x, b)

            # real image
            real_validity, real_sensitive_attribute = self.discriminator(x)

            # fake image
            fake_validity, fake_sensitive_attribute = self.discriminator(x_prime.detach())

            # gradient penalty
            gradient_penalty = self.gradient_penalty(x, x_prime)

            # Adversarial loss
            loss_adv_D = -torch.mean(real_validity) +torch.mean(fake_validity) + lambda_gp * gradient_penalty

            # attribute classification loss
            loss_attr_C = self.bcewithlogits(real_sensitive_attribute, a.float())

            loss_total_D_C = loss_adv_D + lambda_attr_c * loss_attr_C


            self.log('loss_adv_D', loss_adv_D, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_attr_C', loss_attr_C, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_total_D_C', loss_total_D_C, on_step=True, on_epoch=True, prog_bar=True)


            return loss_total_D_C


    def validation_step(self, batch, batch_idx, optimizer_idx):
        pass

    def test_step(self, batch, batch_idx):
        img_1, img_2, match = batch
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
        torch.save(RAPP_confusion_cos, 'RAPP_cos.pt')



def train():
    celeba_data_module = CelebaRAPPDatasetInterface(num_workers=2,
                                  dataset='celeba_data',
                                  batch_size=16,
                                  dim_img=224,
                                  data_dir='E:\datasets\celeba',  # 'D:\datasets\celeba'
                                  sensitive_dim=1,
                                  identity_nums=10177,
                                  pin_memory=True)

    lfw_data_module = LFWInterface(num_workers=0,
                                   dataset='lfw',
                                   data_dir='E:\datasets\lfw\lfw112',
                                   batch_size=256,
                                   dim_img=224,
                                   sensitive_attr='Male',
                                   purpose='face_recognition',
                                   pin_memory=False,
                                   identity_nums=5749,
                                   sensitive_dim=1)

    adience_data_module = AdienceInterface(num_workers=0,
                                           dataset='adience',
                                           data_dir='E:\datasets\Adience',
                                           batch_size=256,
                                           dim_img=224,
                                           sensitive_attr='Male',
                                           purpose='face_recognition',
                                           pin_memory=False,
                                           identity_nums=5749,
                                           sensitive_dim=1)


    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/RAPP_experiments/checkpoints/')

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH, name='RAPP_logger')

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model'),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model'),  # Where to save models
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=50,
        precision=32,
        enable_checkpointing=True,
        fast_dev_run=False,
        min_epochs=16,
        max_epochs=16
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    RAPP_model = RAPP()

    resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    print('Model will be created')
    trainer.fit(RAPP_model, celeba_data_module)
    trainer.test(RAPP_model, celeba_data_module)








if __name__ == '__main__':
    x = torch.rand(size=(16, 3, 128, 128))
    s = torch.rand(size=(16, 10))
    #RAPP_net = RAPP_experiments()
    #output1, output2, output3 = RAPP_net(x,s)
    #print(output1.size())

    #generator = Generator()
    #out = generator(x, s)
    #print(out.size())

    #discriminator = Discriminator()
    #print(discriminator)
    #out1, out2  = discriminator(x)
    #print(out1.size())
    #print(out2.size())

    #input1 = torch.tensor([0,0,1,1])
    #input2 = torch.tensor([0,1,0,1])
    #output = xor(input1, input2)
    #print(output)

    # 这个拿去做b向量
    #vector_length = 40
    #pattern = torch.tensor([0,1,0,1])
    #vector = torch.cat([pattern[i % 4].unsqueeze(0) for i in range(vector_length)])
    #vector = vector.to(torch.int32)
    #print(vector)

    #print(pattern())

    #face_match_net = FaceMatch()
    #output = face_match_net(x)
    #print(output.size())


    #RAPP_net = RAPP_experiments()
    #face_representation, real, sensitive_attribute = RAPP_net(x, s)
    #print(face_representation.size())
    #print(real.size())
    #bce = nn.BCEWithLogitsLoss()
    #result = bce(sensitive_attribute, s)
    #print(result)





    train()





