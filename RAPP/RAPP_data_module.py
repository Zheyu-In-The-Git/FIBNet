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





class CelebaRAPPMineGenderData(data.Dataset):
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

        self.male_attr = ['Male']

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

        target_attr = attr[self.male_attr]

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

        self.a = torch.as_tensor(sensitive_attr[mask].values)
        self.a = torch.div(self.a + 1, 2, rounding_mode='floor')
        self.a = self.a.to(torch.float32)

        self.s = torch.as_tensor(target_attr[mask].value)
        self.s = torch.div(self.s + 1, 2, rounding_mode='floor')
        self.s = self.s.to(torch.float32)


    def __len__(self):
        return len(self.filename)


    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba_mtcnn", self.filename[index]))

        #X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.filename[index]))

        x = self.trans(X)

        u = self.u[index, 0] - 1

        a = self.a[index, :]
        a = a.to(torch.int32)

        s = self.s[index, :]
        s = s.to(torch.int32)

        return x, u, a, s

# TODO:种族还没写完
class CelebaRAPPMineRaceData(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 data_dir: str,
                 sensitive_dim: int,
                 identity_nums: int,
                 split: str,  # train, valid, test
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_dim = sensitive_dim
        self.identity_nums = identity_nums
        self.split = split

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
            'train_63%': 'train_63%',
            'valid_7%': 'valid_7%',
            'test_30%': 'test_30%'
        }

        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all", 'train_63%', 'valid_7%', 'test_30%'))]

        fn = partial(os.path.join, self.data_dir)  # csv检索用的
        imgpath_id = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img_path','id'])
        imgpath_white = pd.read_csv(fn('celeba_imgpath_race.csv')).drop(labels='Unnamed: 0', axis=1)
        imgpath_race_id = pd.merge(imgpath_white, imgpath_id, on='img_path', how='left') # 得把其他敏感属性考虑进去

        if split_ == 'train_63%':
            mask = slice(0, 118063, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id[mask]

        elif split_ == 'valid_7%':
            mask = slice(118063, 131207, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id[mask]

        elif split_ == 'test_30%':
            mask = slice(131207, 187644, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id[mask]

        self.dataset_values = self.dataset.values #



