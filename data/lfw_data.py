import torch
import os
import numpy as np
import pandas
import PIL
import torch.utils.data as data
from functools import partial

from torchvision import transforms

from torch.utils.data import DataLoader

import mat73

class LFWData(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int, # 5749？
                 sensitive_attr: str,
                 img_path_replace : bool,
                 split:str
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr
        self.identity_nums = identity_nums

        fn = partial(os.path.join, self.data_dir)
        self.lfw_dataset = pandas.read_csv(fn('lfw_att_40.csv'))

        lfw_dataset_load_indices_train_test = mat73.loadmat(fn('indices_train_test.mat'))

        self.lfw_dataset_id = pandas.read_csv(fn('lfw_train_test_id.csv'), index_col='name')

        if split == 'train':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_train']
        elif split == 'test':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_test']
        elif split == 'all':
            self.lfw_dataset_indices = np.append(lfw_dataset_load_indices_train_test['indices_img_train'],lfw_dataset_load_indices_train_test['indices_img_test'])
        else:
            'please input the correct lfw dataset split string'


        self.lfw_dataset_img_path = self.lfw_dataset.iloc[:,0] # 路径这里需要重新思考

        self.trans = transforms.Compose([
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

        self.img_path_replace = img_path_replace

    def __len__(self):
        return len(self.lfw_dataset_indices)

    def __getitem__(self, index):

        indices = self.lfw_dataset_indices[index] - 1.0 # 对标到索引

        img_path_name = self.lfw_dataset_img_path[indices]

        # print(img_path_name)

        img_name = img_path_name.split('\\')
        img_name = img_name[0]
        u = self.lfw_dataset_id.loc[img_name]['face_id'] - 1
        u = torch.tensor(u)
        u = u.long()


        if self.img_path_replace:
            img_path_name = img_path_name.replace('\\', '/')
        else:
            img_path_name = img_path_name

        x = PIL.Image.open(os.path.join(self.data_dir, "img/", img_path_name))
        x = self.trans(x)


        # u = 0 # 身份需要后面再做

        s = self.lfw_dataset[self.sensitive_attr][indices]
        s = torch.tensor(s).to(torch.float32)
        return x, u, s




class LFWRecognitionTestPairs(data.Dataset):
    def __init__(self, dim_img : int, data_dir : str, img_path_replace : bool):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.data_dir = data_dir
        self.dim_img = dim_img
        self.img_path_replace = img_path_replace

        fn = partial(os.path.join, self.data_dir)
        self.lfw_test_dataset = pandas.read_csv(fn('lfw_test_pair.txt'), delim_whitespace=True, header=None, index_col=None)

        self.lfw_test_dataset.columns = ['img_x', 'img_y', 'match']

        # 图像变换成张量
        self.trans = transforms.Compose([
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

    def __len__(self):
        return self.lfw_test_dataset.shape[0]

    def __getitem__(self, index):

        img_x_path_name = self.lfw_test_dataset['img_x'][index]
        img_y_path_name = self.lfw_test_dataset['img_y'][index]

        if self.img_path_replace:
            img_x_path_name = img_x_path_name.replace('/', '\\')
            img_y_path_name = img_y_path_name.replace('/', '\\')
        else:
            img_x_path_name = img_x_path_name
            img_y_path_name = img_y_path_name


        img_x = PIL.Image.open(os.path.join(self.data_dir, 'img' ,img_x_path_name))

        img_x = self.trans(img_x)

        img_y = PIL.Image.open(os.path.join(self.data_dir, "img", img_y_path_name))

        img_y = self.trans(img_y)

        match = torch.tensor(self.lfw_test_dataset['match'][index])

        return img_x, img_y, match




if __name__ == '__main__':
    data_dir = '/Users/xiaozhe/datasets/lfw/lfw112'
    loader = LFWData(dim_img=112, data_dir=data_dir, identity_nums=5749, sensitive_attr='Male', img_path_replace=True, split='train')
    train_loader = DataLoader(loader, batch_size=3, shuffle=False)

    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break




    '''
    
    loader_face_recognition = LFWRecognitionTestPairs(dim_img=224, data_dir=data_dir, img_path_replace=False)
    test_loader = DataLoader(loader_face_recognition, batch_size=4, shuffle=False)
    for i, item in enumerate(test_loader):
        print('i', i)
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break
    '''


















