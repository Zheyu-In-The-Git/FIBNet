import torch
import pandas as pd
import torch
import os
import numpy as np
import pandas
import PIL
import torch.utils.data as data
from functools import partial
from facenet_pytorch import MTCNN
import torchvision.transforms.functional as F

from torchvision import transforms
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl

import platform

######################################################################
###################### CelebA  #######################################
######################################################################

class CelebaRAPPData(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int,
                 split: str,  # train, valid, test
                 ):

        self.identity_nums = identity_nums
        self.dim_img = dim_img
        self.data_dir = data_dir

        # 数据集分区
        self.split = split


        self.sensitive_attr = ['Male', 'Young', 'Pale_Skin', 'Bushy_Eyebrows', 'Mouth_Slightly_Open',
                               'Narrow_Eyes', 'Bags_Under_Eyes', 'Black_Hair', 'Mustache', 'Big_Nose']

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
            'train_63%': 'train_63%',
            'valid_7%': 'valid_7%',
            'test_30%': 'test_30%'
        }

        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", 'train_63%', 'valid_7%', 'test_30%'))]

        fn = partial(os.path.join, self.data_dir)  # csv检索用的
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        #sensitive_attr = attr[self.sensitive_attr]

        sensitive_attr = attr[self.sensitive_attr]
        print(sensitive_attr)

        if split_ == 'train_63%':
            mask = slice(0, 127638, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        elif split_ == 'valid_7%':
            mask = slice(127638, 141819, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        elif split_ == 'test_30%':
            mask = slice(141819, 202599, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        elif split_ is None:
            mask = slice(None)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            mask = (splits[1] == split_)  # 后面train test之类的再说吧
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.filename = splits[mask].index.values

        self.u = torch.as_tensor(identity[mask].values)

        self.s = torch.as_tensor(sensitive_attr[mask].values)
        self.s = torch.div(self.s + 1, 2, rounding_mode='floor')
        self.s = self.s.to(torch.float32)


    def __len__(self):
        return len(self.filename)


    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba_mtcnn", self.filename[index]))

        #X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.filename[index]))

        x = self.trans(X)

        u = self.u[index, 0] - 1

        s = self.s[index, :]
        s = s.to(torch.int32)

        return x, u, s




class CelebaRecognitionTestDataSet(data.Dataset):
    def __init__(self, dim_img : int, data_dir : str):
        # Set all input args as attributes
        self.dim_img = dim_img
        self.data_dir = data_dir

        fn = partial(os.path.join, self.data_dir)
        self.celeba_test_dataset = pandas.read_csv(fn('celeba_face_verify_test_dataset.csv'), sep=',')
        print(self.celeba_test_dataset)

        # 图像变换成张量
        self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return self.celeba_test_dataset.shape[0]

    def __getitem__(self, index):
        img_x = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba_mtcnn', self.celeba_test_dataset['img_x'][index]))
        img_x = self.trans(img_x)

        img_y = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba_mtcnn', self.celeba_test_dataset['img_y'][index]))
        img_y = self.trans(img_y)

        match = torch.tensor(self.celeba_test_dataset['match'][index])
        return img_x, img_y, match



class CelebaRAPPDatasetInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers : int,
                 dataset: str,
                 batch_size : int,
                 dim_img : int,
                 data_dir : str,
                 identity_nums : int,
                 **kargs):
        super(CelebaRAPPDatasetInterface).__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = batch_size

        # 数据集相关的参数
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums
        self.pin_memory = kargs['pin_memory']

        self.prepare_data_per_node = True

        self.allow_zero_length_dataloader_with_multiple_devices = True

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_dataset = CelebaRAPPData(dim_img=self.dim_img,
                                             data_dir=self.data_dir,
                                             identity_nums=self.identity_nums,
                                             split='train_63%') # 63%

            self.Train_Dataset = train_dataset

        if stage == 'test' or stage is None:
            self.Test_Dataset = CelebaRecognitionTestDataSet(dim_img=self.dim_img,data_dir=self.data_dir)


    def train_dataloader(self):
        # sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset) * 20)
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)


    def test_dataloader(self):
        return DataLoader(self.Test_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)





if __name__ == '__main__':
    data_dir = 'D:\celeba'
    loader = CelebaRAPPData(dim_img=112, data_dir=data_dir,  identity_nums=10177, split='train_63%')
    train_loader = DataLoader(loader, batch_size=2)
    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break

    dataloader = CelebaRAPPDatasetInterface(dim_img=224, dataset='celeba_data', data_dir=data_dir,
                                     identity_nums=10177, batch_size=2,  pin_memory=False, num_workers=1)
    dataloader.setup(stage='test')
    for i, item in enumerate(dataloader.test_dataloader()):
        print('i', i)
        x, u, s = item
        # print(x)
        # print(u)
        print(s.dtype)
        break




