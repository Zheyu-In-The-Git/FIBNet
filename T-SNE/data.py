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



class CelebaTSNEExperiment(data.Dataset):
    def __init__(self, dim_img:int, data_dir:str, sensitive_attr:str, split:str):
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.split = split
        self.sensitive_attr = sensitive_attr

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
            'train_valid_70%': 'train_valid_70%',
            'test_30%': 'test_30%'
        }

        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", 'train_valid_70%', 'test_30%'))]

        # 确认传入敏感属性类型
        if isinstance(sensitive_attr, list):
            self.sensitive_attr = sensitive_attr
        else:
            self.sensitive_attr = [sensitive_attr]

        fn = partial(os.path.join, self.data_dir)  # csv检索用的
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        sensitive_attr = attr[self.sensitive_attr]

        if split_ == 'train_valid_70%':
            mask = slice(0, 141819, 1)
        elif split_ == 'test_30%':
            mask = slice(141819, 202599, 1)
        elif split_ is None:
            mask = slice(None)
        else:
            mask = (splits[1] == split_)

        male_select_condition = self.sensitive_attr[0] + ' == 1'
        male_dataset = sensitive_attr[mask].query(male_select_condition).head(500)

        female_select_condition = self.sensitive_attr[0] + ' == -1'
        female_dataset = sensitive_attr[mask].query(female_select_condition).head(500)

        self.dataset = pd.concat([male_dataset, female_dataset])
        #print(self.dataset.values[0])

        self.id = identity

        self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset.index.values[index]
        X = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba_mtcnn', img_path))

        x = self.trans(X)

        u = self.id.loc[img_path].values
        u = torch.as_tensor(u) - 1.0
        u = u.long()

        s = torch.as_tensor(self.dataset.values[index])
        s = torch.div(s + 1, 2, rounding_mode='floor')
        s = s.to(torch.float32)
        return x, u, s


class LFWTSNEExperiment(data.Dataset):
    pass

class AdienceTSNEExperiment(data.Dataset):
    pass




