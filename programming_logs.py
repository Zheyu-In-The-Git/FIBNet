

'''

# 查模型大小
def main(args):

    pl.seed_everything(args.seed)


    # 模型加载路径
    load_path = load_model_path_by_args(args) # 测试完成 TODO:会不会需要额外写预训练的路径
    data_module = CelebaInterface(**vars(args)) # 测试完成

    if load_path is None:
        bottlenecknets = ConstructBottleneckNets(args)# 测试完成
    else:
        bottlenecknets = ConstructBottleneckNets(args)# 测试完成
        args.ckpt_path = load_path

    data_module.setup(stage='fit')
    for i, item in enumerate(data_module.train_dataloader()):
        x, u, s = item
        features = x
        break

    model_size = pl.utilities.memory.get_model_size_mb(bottlenecknets)
    print('model_size = {}'.format(model_size))
    bottlenecknets.example_input_array = [features]
    summary = pl.utilities.model_summary.ModelSummary(bottlenecknets, max_depth = -1)
    print(summary)
'''

# Bottleneck_nets.py 4月3日
# 原始bottleneck_nets 的训练方法


'''
import itertools
import importlib
import inspect


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, model_checkpoint


import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter


import pandas as pd
import numpy as np
import torch.nn.functional as F



# 一些准确率，错误率的方法
def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class BottleneckNets(pl.LightningModule):
    def __init__(self, model_name, encoder, decoder, uncertainty_model, sensitive_discriminator, utility_discriminator, lam = 0.0001, gamma = 0.0001, lr = 0.0001, **kwargs):
        super(BottleneckNets, self).__init__()

        self.save_hyperparameters()
        self.model_name = model_name

        # 小网络
        self.encoder = encoder # x->z
        self.decoder = decoder # z->u 向量
        self.uncertainty_decoder = uncertainty_model # z->s 向量 要选最大的一个
        self.sensitive_discriminator = sensitive_discriminator # s->0/1
        self.utility_discriminator = utility_discriminator # u->0/1

        # 超参数设置
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.batch_size = kwargs['batch_size']
        self.identity_nums = kwargs['identity_nums']

        # 设置一些激活函数
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()


        # 监督训练结果

        # 打开计算图
        self.example_input_array = torch.randn([self.batch_size, 3, 224, 224])


    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # 确实变成了标准差
        std = torch.clamp(std, min = 1e-4)
        jitter = 1e-4
        eps = torch.randn_like(std)
        z = mu + eps * (std+jitter)

        return z

    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.softmax(self.decoder(z))
        s_hat = self.sigmoid(self.uncertainty_decoder(z))

        u_value = self.sigmoid(self.utility_discriminator(u_hat))
        s_value = self.sigmoid(self.sensitive_discriminator(s_hat))

        return z, u_hat, s_hat, u_value, s_value, mu, log_var

    def configure_loss(self, pred, true, loss_type):
        if loss_type == 'MSE':
            mse = nn.MSELoss()
            return mse(pred, true)
        elif loss_type == 'BCE':
            bce = nn.BCEWithLogitsLoss()
            return bce(pred, true)
        elif loss_type == 'CE':
            ce = nn.CrossEntropyLoss()
            return ce(pred, true)
        else:
            raise ValueError("Invalid Loss Type!")



    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999

        opt_phi_theta_xi = optim.Adam(itertools.chain(self.encoder.parameters(),
                                                      self.decoder.parameters(),
                                                      self.uncertainty_decoder.parameters()),
                                                      lr=self.lr, betas=(b1, b2))

        opt_omiga = optim.Adam(self.utility_discriminator.parameters(), lr = self.lr, betas=(b1, b2))

        opt_tao = optim.Adam(self.sensitive_discriminator.parameters(), lr=self.lr, betas=(b1, b2))


        return [opt_phi_theta_xi, opt_omiga, opt_tao, opt_phi_theta_xi], []

    def loss_fn_KL(self, mu, log_var):
        loss = 0.5 * (torch.pow(mu, 2) + torch.exp(log_var) - log_var - 1).sum(1).mean()
        return loss

    def loss_adversarially(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss()
        return loss(y_hat, y)

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(preds, labels.cpu().detach().numpy())
        return accuracy, misclass_rate

    def kl_estimate_value(self, discriminating):
        discriminated = self.sigmoid(discriminating) + 1e-4
        kl_estimate_value = (torch.log(discriminated) - torch.log(1-discriminated)).sum(1).mean()
        return kl_estimate_value.detach()

    def training_step(self, batch, batch_idx, optimizer_idx):
        ###################
        # data processing #
        ###################
        x, u, s = batch

        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        u_hat = self.decoder(z)
        s_hat = self.uncertainty_decoder(z)

        ################################################################
        # training the encoder, utility decoder and uncertainty decoder#
        ################################################################

        if optimizer_idx == 0:
            loss_phi_theta_xi = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u, 'CE') - self.lam * self.configure_loss(self.sigmoid(s_hat), s, 'MSE')
            tensorboard_log = {'loss_phi_theta_xi': loss_phi_theta_xi.detach()}
            self.log('KL_divergence', self.loss_fn_KL(mu, log_var), prog_bar=True, logger=True, on_step=True)
            self.log('cross_entropy', self.configure_loss(u_hat, u, 'CE'), prog_bar=True, logger=True, on_step=True)
            self.log('mean_square_error', self.configure_loss(s_hat, s, 'MSE'), prog_bar=True, logger=True, on_step=True)
            self.log('loss_phi_theta_xi', loss_phi_theta_xi, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return  {'loss': loss_phi_theta_xi, 'log':tensorboard_log}

        #################################
        ## training the u_discriminator##
        #################################
        if optimizer_idx == 1:

            u_valid = torch.ones(u.size(0), 1)
            u_valid = u_valid.type_as(u)
            u_valid = u_valid.to(torch.float32)

            u_one_hot = F.one_hot(u, num_classes = self.identity_nums)
            real_u_discriminator_value = self.utility_discriminator(u_one_hot.to(torch.float32))
            loss_real = self.configure_loss(real_u_discriminator_value, u_valid, 'BCE')


            u_fake = torch.zeros(u.size(0), 1)
            u_fake = u_fake.type_as(u)
            u_fake = u_fake.to(torch.float32)

            fake_u_discrimination_value = self.utility_discriminator(u_hat.detach())
            loss_fake = self.configure_loss(fake_u_discrimination_value, u_fake, 'BCE')


            loss_omiga = (loss_real + loss_fake) * self.gamma *0.5
            tensorboard_log = {'loss_omiga':loss_omiga.detach()}
            self.log('loss_omiga',loss_omiga, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return {'loss':loss_omiga, 'log':tensorboard_log}


        #################################
        ## training the s_discriminator##
        #################################
        if optimizer_idx == 2:
            s_valid = torch.ones(s.size(0), 1)
            s_valid = s_valid.type_as(s)
            real_s_discriminator_value = self.sensitive_discriminator(s)
            real_s_loss = self.configure_loss(real_s_discriminator_value, s_valid, 'BCE')

            s_fake = torch.zeros(s.size(0), 1)
            s_fake = s_fake.type_as(s)
            fake_s_discrimination_value = self.sensitive_discriminator(s_hat.detach())
            fake_s_loss = self.configure_loss(fake_s_discrimination_value, s_fake, 'BCE')

            loss_tao = - self.gamma * (real_s_loss + fake_s_loss) * 0.5


            tensorboard_log = {'loss_tao': loss_tao.detach()}
            self.log('loss_tao', loss_tao, prog_bar=True, logger=True, on_step=True, on_epoch = True)


            return {'loss':loss_tao, 'log':tensorboard_log}


        ######################################################################
        ## training the utility decoder and uncertainty decoder adversarially##
        ######################################################################
        if optimizer_idx == 3:
            u_valid = torch.ones(u.size(0), 1)
            u_valid = u_valid.type_as(u)
            u_valid = u_valid.to(torch.float32)

            s_valid = torch.ones(s.size(0), 1)
            s_valid = s_valid.type_as(s)
            s_valid = s_valid.to(torch.float32)


            #对抗损失
            loss_adversarial_phi_theta_xi = self.gamma * self.loss_adversarially(self.utility_discriminator(self.decoder(z)),u_valid) \
                                                 - self.lam * self.loss_adversarially(self.sensitive_discriminator(self.uncertainty_decoder(z)), s_valid)


            # 识别准确率
            u_accuracy, u_misclass_rate = self.get_stats(self.softmax(self.decoder(z)), u)
            s_accuracy, s_misclass_rate = self.get_stats(self.sigmoid(self.uncertainty_decoder(z)), s)


            train_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u, 'CE') - self.lam * self.configure_loss(self.sigmoid(s_hat), s, 'MSE') + \
                               self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

            tensorboard_log = {'loss_adversarial_phi_theta_xi': loss_adversarial_phi_theta_xi.detach(),
                               'train_u_accuracy': u_accuracy,
                               'train_u_error_rate': u_misclass_rate,
                               'train_s_accuracy': s_accuracy,
                               'train_s_error_rate': s_misclass_rate,
                               'train_total_loss': train_loss_total.detach()}

            self.log('loss_adversarial_phi_theta_xi', loss_adversarial_phi_theta_xi, prog_bar=True, logger=True, on_step=True, on_epoch = True)
            self.log_dict(tensorboard_log, prog_bar=True, logger=True, on_step=True, on_epoch = True)

            return {'loss':loss_adversarial_phi_theta_xi, 'log': tensorboard_log}

    # 验证步骤
    def validation_step(self, batch, batch_idx):

        # 数据
        x, u, s = batch
        print(batch_idx)

        z, u_hat, s_hat, u_value, s_value, mu, log_var =self.forward(x)


        val_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u,'CE') - self.lam * self.configure_loss(self.sigmoid(s_hat), s, 'MSE') + \
                           self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

        u_accuracy, u_misclass_rate = self.get_stats(u_hat, u)
        s_accuracy, s_misclass_rate = self.get_stats(s_hat, s)

        tensorboard_logs = {'val_loss_total': val_loss_total,
                            'val_u_accuracy': u_accuracy,
                            'val_u_misclass_rate': u_misclass_rate,
                            'val_s_accuracy':s_accuracy,
                            'val_s_misclass_rate': s_misclass_rate
                            }

        self.log_dict(tensorboard_logs, prog_bar=True, logger=True, on_step=True, on_epoch = True)
        return {'val_loss_total': val_loss_total, 'val_u_accuracy': u_accuracy}


    def test_step(self, batch, batch_idx):
        # 数据
        x, u, s = batch

        z, u_hat, s_hat, u_value, s_value, mu, log_var = self.forward(x)

        test_loss_total = self.loss_fn_KL(mu, log_var) + self.gamma * self.configure_loss(u_hat, u,'CE') - self.lam * self.configure_loss(self.sigmoid(s_hat), s, 'MSE') + \
                         self.gamma * self.kl_estimate_value(self.utility_discriminator(self.decoder(z))) - self.lam * self.kl_estimate_value(self.sensitive_discriminator(self.uncertainty_decoder(z)))

        u_accuracy, u_misclass_rate = self.get_stats(u_hat, u)
        s_accuracy, s_misclass_rate = self.get_stats(s_hat, s)

        tensorboard_logs = {'test_loss_total': test_loss_total,
                            'test_u_accuracy': u_accuracy,
                            'test_u_misclass_rate': u_misclass_rate,
                            'test_s_accuracy': s_accuracy,
                            'test_s_misclass_rate': s_misclass_rate
                            }
        self.log_dict(tensorboard_logs, prog_bar=True, logger=True, on_step=True, on_epoch = True)


'''



# main.py 4月3日
# 原始main.py文件

'''
import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import ConstructBottleneckNets
from data import CelebaInterface
from utils import load_model_path_by_args


def load_callbacks(load_path):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_u_accuracy_epoch',
        mode='max',
        patience=10,
        min_delta=0.001,
    ))

    callbacks.append(plc.ModelCheckpoint(
        save_last=True,
        dirpath = os.path.join(load_path, 'saved_models'),
        every_n_train_steps=50,
    ))

    callbacks.append(
        plc.LearningRateMonitor('epoch')
    )
    return callbacks



# 构建瓶颈网络模型
def main(args):

    pl.seed_everything(args.seed)

    # 模型加载路径，
    load_path = load_model_path_by_args(args)
    print(load_path) # lightning_logs\bottleneck_test_version_1\checkpoints

    data_module = CelebaInterface(**vars(args))

    logger = TensorBoardLogger(save_dir=load_path + args.log_dir, name=args.log_name, version='version_1',)  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    trainer = Trainer(
        default_root_dir=os.path.join(load_path, 'saved_models'),
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=load_callbacks(load_path),
        logger=logger,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        accelerator='gpu',
        devices=1,
        check_val_every_n_epoch=30,
        #fast_dev_run=100
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = True

    bottlenecknets = ConstructBottleneckNets(args)


    if args.RESUME:
        # 模型重载训练阶段
        resume_checkpoint_dir = os.path.join(load_path, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, args.ckpt_name)
        print('Found pretrained model at ' + resume_checkpoint_path + ', loading ... ')  # 重新加载
        model = bottlenecknets
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint_path)
    else:
        # 模型创建阶段
        resume_checkpoint_dir = os.path.join(load_path, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, args.ckpt_name)
        print('Model will be created')
        model = bottlenecknets
        trainer.fit(model, datamodule=data_module)
        trainer.save_checkpoint(resume_checkpoint_path)

if __name__ == '__main__':
    # 设置GPU，使得代码能够复现
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    # 设置运行芯片
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    # Create checkpoint path if it doesn't exist yet

    # 数据集的路径 CELEBA的位置需要更改
    DATASET_PATH = 'D:\datasets\celeba' # D:\datasets\celeba

    # tensorboard记录
    LOG_PATH = os.environ.get('LOG_PATH', '\lightning_logs')
    # 模型加载与命名
    VERSION = 'bottleneck_experiment_latent512_lam00001_gamma1000'
    VERSION_NUM = '_1/'
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/' + VERSION + VERSION_NUM + 'checkpoints/')

    ###################
    ## 设置参数这里开始 #
    ###################

    parser = ArgumentParser()

    # 预训练模型路径加载
    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default = CHECKPOINT_PATH, type=str, help = 'The root directory of checkpoints.')
    parser.add_argument('--load_ver', default='bottleneck_experiment_latent512_lam00001_gamma1000', type=str, help = '训练和加载模型的命名 采用')
    parser.add_argument('--load_v_num', default = 1, type=int)
    parser.add_argument('--RESUME', default=False, type=bool, help = '是否需要重载模型')
    parser.add_argument('--ckpt_name', default='bottleneck_experiment_latent512_lam00001_gamma1000.ckpt', type = str )


    #基本超参数，构建小网络的基本参数
    parser.add_argument('--dim_img', default=224, type=int)
    parser.add_argument('--sensitive_dim', default = 1, type = int)
    parser.add_argument('--latent_dim', default = 512, type = int)
    parser.add_argument('--identity_nums', default=10177, type = int)

    # 基本系统参数
    parser.add_argument('--seed', default=43, type = int)

    # 数据集参数设置
    parser.add_argument('--dataset', default='celeba_data', type=str)
    parser.add_argument('--data_dir', default = DATASET_PATH, type=str)
    parser.add_argument('--num_workers', default =2, type=int)
    parser.add_argument('--sensitive_attr', default='Male', type=str)
    parser.add_argument('--pin_memory', default = True)

    # bottleneck_nets的参数
    parser.add_argument('--encoder_model', default='ResNet50',type = str)
    parser.add_argument('--model_name', default='bottleneck_experiment_version', type = str)
    parser.add_argument('--lam', default=0.0001, type = float)
    parser.add_argument('--gamma', default=1000, type=float)
    parser.add_argument('--batch_size', default = 64, type=int)
    parser.add_argument('--max_epochs', default=50, type = int)
    parser.add_argument('--min_epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # 日志参数
    parser.add_argument('--log_dir', default=LOG_PATH, type=str)
    parser.add_argument('--log_name', default='tensorboard_log',type=str)

    args = parser.parse_args()

    main(args)


'''

# construct_bottleneck_nets.py 4月3日 文件
'''
def ConstructBottleneckNets(args, **kwargs):

    if args.encoder_model == 'ResNet50':
        # 编码器
        get_encoder = ResNet50Encoder(latent_dim=args.latent_dim, channels=3, act_fn='PReLU')

        # 解码器
        get_decoder = ResNet50Decoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, act_fn='Softmax')

        # 模糊器
        get_uncertainty_model = UncertaintyModel(latent_dim=args.latent_dim, sensitive_dim=args.sensitive_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)

        # s判别器
        get_sensitive_discriminator = SensitiveDiscriminator(sensitive_dim=args.sensitive_dim)

    elif args.encoder_model == 'ResNet101':
        # 编码器
        get_encoder = ResNet101Encoder(latent_dim=args.latent_dim, channels=3, act_fn='PReLU')

        # 解码器
        get_decoder = ResNet101Decoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, act_fn='Softmax')

        # 模糊器
        get_uncertainty_model = UncertaintyModel(latent_dim=args.latent_dim, sensitive_dim=args.sensitive_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)

        # s判别器
        get_sensitive_discriminator = SensitiveDiscriminator(sensitive_dim=args.sensitive_dim)


    elif args.encoder_model == 'ResNet18':
        # 编码器
        get_encoder = ResNet18Encoder(latent_dim=args.latent_dim, channels=3, act_fn='PReLU')

        # 解码器
        get_decoder = ResNet18Decoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, act_fn='Softmax')

        # 模糊器
        get_uncertainty_model = UncertaintyModel(latent_dim=args.latent_dim, sensitive_dim=args.sensitive_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)

        # s判别器
        get_sensitive_discriminator = SensitiveDiscriminator(sensitive_dim=args.sensitive_dim)

    elif args.encoder_model == 'LitModel':
        # 编码器
        get_encoder = LitEncoder1(latent_dim=args.latent_dim, channels=3, act_fn='PReLU')

        # 解码器
        get_decoder = LitDecoder1(latent_dim=args.latent_dim, identity_nums=args.identity_nums, act_fn='Softmax')

        # 模糊器
        get_uncertainty_model = UncertaintyModel(latent_dim=args.latent_dim, sensitive_dim=args.sensitive_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)

        # s判别器
        get_sensitive_discriminator = SensitiveDiscriminator(sensitive_dim=args.sensitive_dim)




    return BottleneckNets(model_name= args.model_name, encoder=get_encoder, decoder=get_decoder,
                          uncertainty_model= get_uncertainty_model,
                          utility_discriminator= get_utility_discriminator,
                          sensitive_discriminator = get_sensitive_discriminator,
                          batch_size=args.batch_size, lam = args.lam, gamma=args.gamma, lr = args.lr, identity_nums = args.identity_nums)



'''


from torchmetrics.classification import ConfusionMatrix # 混淆矩阵


'''
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        print('one-hot',one_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
'''