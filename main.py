import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from data import CelebaInterface
from utils import load_model_path_by_args
from model import Encoder, Decoder, LatentDiscriminator, UtilityDiscriminator, BottleneckNets
from arcface_resnet50 import ArcfaceResnet50


def load_callbacks(load_path):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_u_accuracy',
        mode='max',
        patience=3,
        min_delta=0.001,
    ))

    callbacks.append(plc.ModelCheckpoint(
        save_last=True,
        dirpath = os.path.join(load_path, 'saved_models'),
        every_n_train_steps=50,
        monitor='val_u_accuracy',  # 用valid_acc比较好
        mode='max'  # 用valid_acc比较好 保存top_k 3个比较好把
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
    print(load_path)

    #数据模块
    data_module = CelebaInterface(num_workers = args.num_workers,
                 dataset=args.dataset,
                 batch_size=args.batch_size,
                 dim_img=args.dim_img,
                 data_dir=args.data_dir,
                 sensitive_dim=args.sensitive_dim,
                 identity_nums=args.identity_nums,
                 sensitive_attr=args.sensitive_attr,
                 pin_memory=args.pin_memory)

    #打开记录器
    logger = TensorBoardLogger(save_dir=load_path, name='tensorboard_log')  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    # 加载网络模块，并构建BottleneckNets
    arcface_resnet50_net =ArcfaceResnet50(in_features=args.latent_dim, out_features=10177, s=64.0, m=0.50)
    arcface = arcface_resnet50_net.load_from_checkpoint(args.arcface_resnet50_path)
    encoder = Encoder(latent_dim=args.latent_dim, arcface_model=arcface)
    decoder = Decoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, s=64.0, m=0.50, easy_margin=False)

    bottlenecknets = BottleneckNets(model_name=args.model_name, encoder=encoder, decoder=decoder,
                                    beta=args.beta, batch_size=args.batch_size, identity_nums=args.identity_nums)


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
        check_val_every_n_epoch=5,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None


    if args.RESUME:
        # 模型重载训练阶段
        resume_checkpoint_dir = os.path.join(load_path, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, args.ckpt_name)
        print('Found pretrained model at ' + resume_checkpoint_path + ', loading ... ')  # 重新加载
        trainer.fit(bottlenecknets, datamodule=data_module, ckpt_path='lightning_logs/bottleneck_experiment_latent_new_512_beta0.0001/checkpoints/saved_models/last.ckpt')
        trainer.test(bottlenecknets, data_module)
        #trainer.save_checkpoint(resume_checkpoint_path)

    else:
        # 模型创建阶段
        resume_checkpoint_dir = os.path.join(load_path, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, args.ckpt_name)
        print('Model will be created')
        trainer.fit(bottlenecknets, data_module)
        trainer.test(bottlenecknets, data_module)
        trainer.save_checkpoint(resume_checkpoint_path)


if __name__ == '__main__':
    '''
    设置各类系统参数
    '''
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
    VERSION = 'bottleneck_experiment_latent_new_512_beta0.001'
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/' + VERSION + '/checkpoints/')

    ###################
    ## 设置参数这里开始 #
    ###################

    parser = ArgumentParser()

    # 预训练模型路径加载
    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default = CHECKPOINT_PATH, type=str, help = 'The root directory of checkpoints.')
    parser.add_argument('--load_ver', default='bottleneck_experiment_latent512_beta0.001', type=str, help = '训练和加载模型的命名 采用')
    parser.add_argument('--load_v_num', default = 1, type=int)
    parser.add_argument('--RESUME', default=False, type=bool, help = '是否需要重载模型')
    parser.add_argument('--ckpt_name', default='bottleneck_experiment_latent512_beta0.001.ckpt', type = str )
    parser.add_argument('--arcface_resnet50_path', default=r'lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=48-step=95800.ckpt')


    #基本超参数，构建小网络的基本参数
    parser.add_argument('--dim_img', default=224, type=int)
    parser.add_argument('--sensitive_dim', default = 1, type = int)
    parser.add_argument('--latent_dim', default = 512, type = int)
    parser.add_argument('--identity_nums', default=10177, type = int)

    # 基本系统参数
    parser.add_argument('--seed', default=83, type = int)

    # 数据集参数设置
    parser.add_argument('--dataset', default='celeba_data', type=str)
    parser.add_argument('--data_dir', default=DATASET_PATH, type=str)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--sensitive_attr', default='Male', type=str)
    parser.add_argument('--pin_memory', default = False)

    # bottleneck_nets的参数
    parser.add_argument('--model_name', default='bottleneck', type=str)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epochs', default=150, type = int)
    parser.add_argument('--min_epochs', default=100, type=int)


    # 日志参数
    parser.add_argument('--log_dir', default=LOG_PATH, type=str)
    parser.add_argument('--log_name', default='tensorboard_log',type=str)

    args = parser.parse_args()

    main(args)
