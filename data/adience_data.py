import pandas as pd
import torch
import os
import numpy as np
import pandas
import PIL
import torch.utils.data as data
from functools import partial

from torchvision import transforms

from torch.utils.data import DataLoader

# 要做训练集和测试集吗


class AdienceData(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 identity_nums:int, # 2284?
                 sensitive_attr:str,
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr
        self.identity_nums = identity_nums

        fn = partial(os.path.join, self.data_dir)
        adience_dataset_fold_0 = pandas.read_table(fn('fold_0_data.txt'), index_col=False)
        adience_dataset_fold_1 = pandas.read_table(fn('fold_1_data.txt'), index_col=False)
        adience_dataset_fold_2 = pandas.read_table(fn('fold_2_data.txt'), index_col=False)
        adience_dataset_fold_3 = pandas.read_table(fn('fold_3_data.txt'), index_col=False)
        adience_dataset_fold_4 = pandas.read_table(fn('fold_4_data.txt'), index_col=False)

        adience_dataset = pd.concat([adience_dataset_fold_0, adience_dataset_fold_1,
                                          adience_dataset_fold_2, adience_dataset_fold_3,
                                          adience_dataset_fold_4],  ignore_index=True)

        adience_dataset = adience_dataset.dropna(subset=['gender'])
        adience_dataset = adience_dataset.reset_index()

        self.adience_dataset = adience_dataset[['user_id', 'original_image', 'face_id', 'gender']]

        self.trans = transforms.Compose([
            transforms.Resize(self.dim_img),
            transforms.CenterCrop((178, 178)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


    def __len__(self):
        return self.adience_dataset.shape[0]

    def __getitem__(self, index):

        user_id_path = self.adience_dataset['user_id'][index]
        original_image = self.adience_dataset['original_image'][index]
        face_id = self.adience_dataset['face_id'][index]
        gender = self.adience_dataset['gender'][index]

        img_sub_path = 'coarse_tilt_aligned_face' + '.' + str(face_id) + '.' + original_image

        img_path = os.path.join(self.data_dir, 'faces', user_id_path, img_sub_path)

        x = PIL.Image.open(img_path)

        x = self.trans(x)

        u = face_id - 1

        u = torch.tensor(u)

        if gender == 'f':
            s = torch.tensor([0.0])
        else:
            s = torch.tensor([1.0])


        return x, u, s

class AdienceRecognitionTestPairs(data.Dataset):
    def __init__(self, dim_img : int, data_dir : str):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.data_dir = data_dir
        self.dim_img = dim_img

        fn = partial(os.path.join, self.data_dir)
        self.adience_dataset_pairs = pandas.read_csv(fn('adience_dataset_facerecognition.csv'), header = 0)
        # print(self.adience_dataset_pairs['img_x'])


        # 图像变换成张量
        self.trans = transforms.Compose([
                                    transforms.CenterCrop((178, 178)),
                                    transforms.Resize(self.dim_img),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])

    def __len__(self):
        return self.adience_dataset_pairs.shape[0]

    def __getitem__(self, index):


        img_x = PIL.Image.open(os.path.join(self.data_dir, 'faces', self.adience_dataset_pairs['img_x'][index]))

        img_x = self.trans(img_x)

        img_y = PIL.Image.open(os.path.join(self.data_dir, 'faces', self.adience_dataset_pairs['img_y'][index]))

        img_y = self.trans(img_y)

        match = torch.tensor(self.adience_dataset_pairs['match'][index])

        return img_x, img_y, match





if __name__ == '__main__':
    data_dir = '/Volumes/xiaozhe_SSD/datasets/Adience'
    loader = AdienceData(dim_img=224, data_dir=data_dir, identity_nums=10177, sensitive_attr='Male')
    train_loader = DataLoader(loader, batch_size=2, shuffle=False)


    '''
    
    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break
    '''
    loader_face_recogntion = AdienceRecognitionTestPairs(dim_img=224, data_dir=data_dir)
    test_pairs = DataLoader(loader_face_recogntion, batch_size=2, shuffle=False)
    for i, item in enumerate(train_loader):
        print('i', i)
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break

















