import torch
import os
import numpy as np
import pandas
import PIL
import pickle as pkl
import torch.utils.data as data
from functools import partial


from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import verify_str_arg

from torch.utils.data import DataLoader

class LFWData(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int, # 1680
                 sensitive_attr: str,
                 img_path_replace : bool
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr
        self.identity_nums = identity_nums

        fn = partial(os.path.join, self.data_dir)
        self.lfw_dataset = pandas.read_csv(fn('lfw_att_40.csv'))

        self.lfw_dataset_img_path = self.lfw_dataset.iloc[:,0]

        self.trans = transforms.Compose([
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

        self.img_path_replace = img_path_replace

    def __len__(self):
        return self.lfw_dataset.shape[0]

    def __getitem__(self, index):

        img_path_name = self.lfw_dataset_img_path[index]

        if self.img_path_replace:
            img_path_name = img_path_name.replace('\\', '/')
        else:
            img_path_name = img_path_name

        x = PIL.Image.open(os.path.join(self.data_dir, "img/", img_path_name))
        x = self.trans(x)

        u = 0

        s = self.lfw_dataset[self.sensitive_attr][index]
        s = torch.tensor(s).to(torch.float32)
        return x, u, s




class LFWRecognitionTestDataSet(data.Dataset):
    def __init__(self, dim_img : int, data_dir : str, img_path_replace : bool):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.data_dir = data_dir
        self.dim_img = dim_img
        self.img_path_replace = img_path_replace

        fn = partial(os.path.join, self.data_dir)
        self.lfw_test_dataset = pandas.read_csv(fn('lfw_test_pair.txt'), delim_whitespace=True, header=None, index_col=0)

        # 图像变换成张量
        self.trans = transforms.Compose([
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

    def __len__(self):
        return self.lfw_test_dataset.shape[0]

    def __getitem__(self, index):




        if self.img_path_replace:
            img_path_name = img_path_name.replace('\\', '/')

        img_x = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba",
                                            self.celeba_test_dataset['img_x'][index]))

        img_x = self.trans(img_x)

        img_y = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba",
                                            self.celeba_test_dataset['img_y'][index]))

        img_y = self.trans(img_y)

        match = torch.tensor(self.celeba_test_dataset['match'][index])

        return img_x, img_y, match




if __name__ == '__main__':
    data_dir = '/Volumes/xiaozhe_SSD/datasets/lfw/lfw112'
    loader = LFWData(dim_img=224, data_dir=data_dir, identity_nums=1680, sensitive_attr='Male', img_path_replace=True)
    train_loader = DataLoader(loader, batch_size=2, shuffle=False)
    
    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break

















