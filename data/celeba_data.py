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
        self.dim_img = dim_img
        self.data_dir = data_dir

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
            'train_63%': 'train_63%',
            'valid_7%': 'valid_7%',
            'test_30%':'test_30%'
        }

        split_ =  split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", 'train_63%', 'valid_7%', 'test_30%'))]


        fn = partial(os.path.join, self.data_dir) # csv检索用的
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        sensitive_attr = attr[self.sensitive_attr]


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
            mask = (splits[1] == split_) # 后面train test之类的再说吧
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


        self.filename = splits[mask].index.values


        self.u = torch.as_tensor(identity[mask].values)

        self.s = torch.as_tensor(sensitive_attr[mask].values)
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
        X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba_mtcnn", self.filename[index]))

        #X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.filename[index]))

        x = self.trans(X)

        u = self.u[index, 0] - 1

        s = self.s[index, :]

        return x, u, s


class CelebaRaceDataset(data.Dataset):
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
        imgpath_race_id = pd.merge(imgpath_white, imgpath_id, on='img_path', how='left')

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




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.data_dir, "img_align_celeba/img_align_celeba_mtcnn", self.dataset['img_path'][index]))
        x = self.trans(X)
        #to_img = transforms.ToPILImage()
        #img = to_img(x)
        #img.show()

        u = self.dataset['id'][index] - 1
        u = torch.tensor(u)
        u = u.long()


        s = self.dataset['white'][index]
        s = torch.tensor([s]).to(torch.float32)

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


if __name__ == '__main__':

    #data_dir = 'D:\datasets\celeba'
    data_dir = '/Users/xiaozhe/datasets/celeba'
    #data_dir = 'D:\celeba'



    '''
    
    loader = CelebaData(dim_img=112, data_dir=data_dir, sensitive_dim=2, identity_nums=10177, sensitive_attr='Male', split='train_63%')
    # dataset = CelebaTSNEExperiment(dim_img=112, data_dir=data_dir, sensitive_attr='Male', split='test_30%')
    train_loader = DataLoader(loader, batch_size=64)
    #print(sampler)


    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(s)
        break
    '''

    loader = CelebaRaceDataset(dim_img=112, data_dir=data_dir, sensitive_dim=1, identity_nums=10177,split='train_63%')
    train_loader = DataLoader(loader, batch_size=2)
    for i, item in enumerate(train_loader):
        print('i', i )
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break






    '''
    
    loader = CelebaRecognitionTestDataSet(dim_img=112, data_dir = data_dir)
    validation_loader = DataLoader(loader, batch_size=2, shuffle=False)
    for i, item in enumerate(validation_loader):
        print('i', i)
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break
    '''




