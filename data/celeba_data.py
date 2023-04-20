

import torch
import os
import numpy as np
import pandas
import PIL
import torch.utils.data as data
from functools import partial


from torchvision import transforms
from torchvision.datasets.utils import verify_str_arg



from torch.utils.data import DataLoader
from torchvision.datasets import CelebA


class CelebaData(data.Dataset):

    base_folder = 'celeba'
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self,
                 dim_img : int,
                 data_dir : str,
                 sensitive_dim : int,
                 identity_nums : int,
                 sensitive_attr: str,
                 split: str, # train, valid, test
                 ):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.sensitive_dim = sensitive_dim
        self.identity_nums = identity_nums

        # 数据集分区
        self.split = split
        # 确认传入敏感属性类型
        if isinstance(sensitive_attr, list):
            self.sensitive_attr = sensitive_attr
        else:
            self.sensitive_attr = [sensitive_attr]


        split_map = {
            'train':0,
            'valid':1,
            'test':2,
            'all':None,
            'train_valid_70%': 'train_valid_70%',
            'test_30%':'test_30%'
        }

        split_ =  split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", 'train_valid_70%', 'test_30%'))]
        # print('split_', split_)


        fn = partial(os.path.join, self.data_dir) # csv检索用的
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        sensitive_attr = attr[self.sensitive_attr]


        # mask = slice(None) if split_ is None else (splits[1] == split_)
        # mask = slice(0, 182637, 1) if split_ == 'train_valid' else (splits[1] == split_) # 将train数据集和val数据集合并成一块

        if split_ == 'train_valid_70%':
            mask = slice(0, 141819, 1)
        elif split_ == 'test_30%':
            mask = slice(141819, 202599, 1)
        elif split_ is None:
            mask = slice(None)
        else:
            mask = (splits[1] == split_)


        self.filename = splits[mask].index.values
        # print(len(self.filename))

        self.u = torch.as_tensor(identity[mask].values)

        self.s = torch.as_tensor(sensitive_attr[mask].values)
        #self.s = (self.s + 1) //2
        self.s = torch.div(self.s +1, 2, rounding_mode='floor')
        self.s = self.s.to(torch.float32)


    def __len__(self):
        return len(self.filename)

    def to_one_hot(self, idx, class_num):
        # 传入tensor，返回tensor
        out = torch.zeros(class_num)
        out[idx] = 1
        return out


    def __getitem__(self, index):

        # 图像
        X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.filename[index]))

        trans = transforms.Compose([transforms.CenterCrop((130, 130)),
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                                    ])

        x = trans(X)


        # 身份信息
        u = self.u[index, 0] - 1
        # u = self.to_one_hot(u, self.identity_nums)
        #u = u.to(torch.float32)

        # 所有属性信息
        s = self.s[index, :]
        # s = self.to_one_hot(s, self.sensitive_dim)


        return x, u, s


class CelebaRecognitionTestDataSet(data.Dataset):
    def __init__(self, dim_img : int, data_dir : str):
        # Set all input args as attributes
        self.__dict__.update(locals())


        self.dim_img = dim_img
        self.data_dir = data_dir

        fn = partial(os.path.join, self.data_dir)
        self.celeba_test_dataset = pandas.read_csv(fn('celeba_face_verify_test_dataset.csv'), sep=',')
        print(self.celeba_test_dataset)

        # 图像变换成张量
        self.trans = transforms.Compose([transforms.CenterCrop((125, 125)),
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

    def __len__(self):
        return self.celeba_test_dataset.shape[0]

    def __getitem__(self, index):
        img_x = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba",
                                            self.celeba_test_dataset['img_x'][index]))

        img_x = self.trans(img_x)




        img_y = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba",
                                            self.celeba_test_dataset['img_y'][index]))

        img_y = self.trans(img_y)

        match = torch.tensor(self.celeba_test_dataset['match'][index])

        return img_x, img_y, match



if __name__ == '__main__':

    #data_dir = '/Users/xiaozhe/PycharmProjects/representation_soft_biometric_enhancement/data/celeba'
    #data_dir = 'D:\datasets\celeba'
    data_dir = '/Volumes/xiaozhe_SSD/datasets/celeba'


    loader = CelebaData(dim_img=224, data_dir=data_dir, sensitive_dim=2, identity_nums=10177, sensitive_attr='Male', split='train_valid_70%')
    train_loader = DataLoader(loader, batch_size=2, shuffle = True)

    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(u, s)
        break
    '''

    loader = CelebaRecognitionTestDataSet(dim_img=224, data_dir = data_dir)
    validation_loader = DataLoader(loader, batch_size=2, shuffle=False)
    for i, item in enumerate(validation_loader):
        print('i', i)
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break
    '''



