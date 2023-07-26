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
import mat73




######################################################################
###################### CelebA  #######################################
######################################################################
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

        self.s = torch.as_tensor(target_attr[mask].values)
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


class CelebaRAPPMineRaceData(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int,
                 split: str,  # train, valid, test
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums
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

        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all", 'train_63%', 'valid_7%', 'test_30%'))]

        fn = partial(os.path.join, self.data_dir)  # csv检索用的
        imgpath_id = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img_path','id'])
        imgpath_white = pd.read_csv(fn('celeba_imgpath_race.csv')).drop(labels='Unnamed: 0', axis=1)
        imgpath_race_id = pd.merge(imgpath_white, imgpath_id, on='img_path', how='left') # 得把其他敏感属性考虑进去
        #print(imgpath_race_id)

        attr = pandas.read_csv(fn('list_attr_celeba.txt'), delim_whitespace=True, header=1)
        sensitive_attr = attr[self.sensitive_attr]
        sensitive_attr = sensitive_attr.reset_index()
        sensitive_attr = sensitive_attr.rename(columns={'index':'img_path'})
        #print(sensitive_attr)

        imgpath_race_id_sensitiveattr = pd.merge(imgpath_race_id, sensitive_attr, on='img_path', how='left')
        print(imgpath_race_id_sensitiveattr) # 主要的数据集在这里

        if split_ == 'train_63%':
            mask = slice(0, 118063, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id_sensitiveattr[mask]

        elif split_ == 'valid_7%':
            mask = slice(118063, 131207, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id_sensitiveattr[mask]

        elif split_ == 'test_30%':
            mask = slice(131207, 187644, 1)
            self.trans = transforms.Compose([transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id_sensitiveattr[mask]

        self.dataset_values = self.dataset.values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba_mtcnn', self.dataset_values[index][0]))

        x = self.trans(X)

        u = self.dataset_values[index][2] - 1
        u = torch.tensor(u)
        u = u.long()

        s = self.dataset_values[index][1]
        s = torch.tensor([s]).to(torch.int32)

        a = self.dataset_values[index][3:]
        a = a.astype('float32')
        a = torch.from_numpy(a)
        a = torch.div(a + 1, 2, rounding_mode='floor')
        a = a.to(torch.int32)
        return x, u, a, s

# 开始写LFW吧


######################################################################
###################### LFW  #######################################
######################################################################


class LFWRAPPMineGenderData(data.Dataset):
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
        #print(self.lfw_dataset.keys())

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

        self.a_attr = ['Male', 'Young', 'Pale_Skin', 'Bushy_Eyebrows', 'Mouth_Slightly_Open',
                               'Narrow_Eyes', 'Bags_Under_Eyes', 'Black_Hair', 'Mustache', 'Big_Nose']

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
        s = torch.tensor([s]).to(torch.int32)
        a = self.lfw_dataset.loc[indices, self.a_attr]
        a = torch.tensor(a).to(torch.int32)

        return x, u, a, s

class LFWRAPPMineRaceData(data.Dataset):
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
        self.lfw_dataset = pandas.read_csv(fn('lfw_att_73.csv'))
        self.lfw_dataset_40 = pandas.read_csv(fn('lfw_att_40.csv'))

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

        self.a_attr = ['Male', 'Young', 'Pale_Skin', 'Bushy_Eyebrows', 'Mouth_Slightly_Open',
                       'Narrow_Eyes', 'Bags_Under_Eyes', 'Black_Hair', 'Mustache', 'Big_Nose']
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
        s = torch.tensor([s]).to(torch.int32)

        a = self.lfw_dataset_40.loc[indices, self.a_attr]
        a = torch.tensor(a).to(torch.int32)

        return x, u, a, s



def pattern():
    vector_length = 10
    pattern = torch.tensor([0, 1, 0, 1])
    vector = torch.cat([pattern[i % 4].unsqueeze(0) for i in range(vector_length)])
    vector = vector.to(torch.int32)
    return vector

######################################################################
###################### Adience  #######################################
######################################################################

class AdienceMineGenderData(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 identity_nums:int, # 2284?
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
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

        self.trans_first = transforms.Compose([transforms.CenterCrop((1250, 1250))])
        self.trans_second = transforms.Compose([transforms.Resize((self.dim_img, self.dim_img)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                     std=[0.5, 0.5, 0.5]),
                                                ])

        self.trans = transforms.Compose([transforms.CenterCrop((250,250)),
                                         transforms.Resize((self.dim_img, self.dim_img)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

        self.a = pattern()


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

        #to_img = transforms.ToPILImage()
        #img = to_img(x)
        #img.show()

        # x = self.trans(x)

        u = face_id - 1

        u = torch.tensor(u)

        if gender == 'f':
            s = torch.tensor([0.0])
            s = s.to(torch.int32)
        else:
            s = torch.tensor([1.0])
            s = s.to(torch.int32)

        a = self.a
        a = a.to(torch.int32)

        return x, u, a, s


class AdienceMineRaceData(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 identity_nums:int, # 2284?
                 ):
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums

        fn = partial(os.path.join, self.data_dir)
        adience_fold_0 = pandas.read_table(fn('fold_0_data.txt'), index_col=False)
        adience_fold_1 = pandas.read_table(fn('fold_1_data.txt'), index_col=False)
        adience_fold_2 = pandas.read_table(fn('fold_2_data.txt'), index_col=False)
        adience_fold_3 = pandas.read_table(fn('fold_3_data.txt'), index_col=False)
        adience_fold_4 = pandas.read_table(fn('fold_4_data.txt'), index_col=False)

        self.adience_dataset = pd.concat([adience_fold_0,adience_fold_1,adience_fold_2,adience_fold_3,adience_fold_4], ignore_index=True)
        #print(self.adience_dataset)

        self.userid_imgsubpath_white_faceid = pandas.read_csv(fn('adience_imgpath_race_id.csv')).drop(labels='Unnamed: 0', axis=1)
        #print(self.userid_imgsubpath_white_faceid)

        self.trans = transforms.Compose([transforms.CenterCrop((300, 300)),
                                         transforms.Resize((self.dim_img, self.dim_img)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.a = pattern()

    def __len__(self):
        return len(self.userid_imgsubpath_white_faceid)

    def __getitem__(self, index):
        user_id_path = self.userid_imgsubpath_white_faceid.loc[index]['user_id']
        img_sub_path = self.userid_imgsubpath_white_faceid.loc[index]['img_sub_path']
        race = self.userid_imgsubpath_white_faceid.loc[index]['white']
        face_id = self.userid_imgsubpath_white_faceid.loc[index]['face_id']

        img_path = os.path.join(self.data_dir, 'faces', user_id_path, img_sub_path)

        x = PIL.Image.open(img_path)

        x = self.trans(x)

        #to_img = transforms.ToPILImage()
        #img = to_img(x)
        #img.show()

        u = face_id - 1

        u = torch.tensor(u)

        s = torch.tensor([race])
        s = s.to(torch.float32)

        a = self.a
        a = a.to(torch.int32)

        return x, u, a, s



if __name__ == '__main__':
    data_dir = 'D:\celeba'
    lfw_data_dir = 'D:\lfw\lfw112'
    adience_data_dir = '/Users/xiaozhe/datasets/Adience'


    '''
    
    loader = CelebaRAPPMineGenderData(dim_img=112, data_dir=data_dir,  identity_nums=10177, split='train_63%')
    # dataset = CelebaTSNEExperiment(dim_img=112, data_dir=data_dir, sensitive_attr='Male', split='test_30%')
    train_loader = DataLoader(loader, batch_size=1)

    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, a, s = item
        print(a)
        print(s)
        break


    celeba_race = CelebaRAPPMineRaceData(dim_img=112, data_dir=data_dir, identity_nums=10177, split='train_63%')
    train_loader_celeba_race = DataLoader(celeba_race, batch_size=1)
    for i, item in enumerate(train_loader_celeba_race):
        print('i', i)
        x, u, a, s = item
        print(a)
        break

    LFW_data = LFWRAPPMineGenderData(dim_img=112, data_dir=lfw_data_dir, identity_nums=2088, split='all', sensitive_attr='Male', img_path_replace=True)
    loader_lfw = DataLoader(LFW_data, batch_size=1)
    for i, item in enumerate(loader_lfw):
        print('i', i)
        x, u, a, s = item
        print(a)
        break
    


    LFW_race_data = LFWRAPPMineRaceData(dim_img=112, data_dir=lfw_data_dir, identity_nums=5000, split='all', sensitive_attr='White',img_path_replace=True)
    loader_lfw_race = DataLoader(LFW_race_data, batch_size=1)
    for i, item in enumerate(loader_lfw_race):
        print('i', i)
        x, u, a, s = item
        print(a)
        break
    '''

    adience_gender_datasets = AdienceMineGenderData(dim_img=112, data_dir=adience_data_dir, identity_nums=2284)
    loader_adience_gender = DataLoader(adience_gender_datasets, batch_size=1)
    for i, item in enumerate(loader_adience_gender):
        print('i', i)
        x, u, a, s = item
        print(a)
        break

    adience_race_datasets = AdienceMineRaceData(dim_img=112, data_dir=adience_data_dir, identity_nums=2284)
    loader_adience_race = DataLoader(adience_race_datasets, batch_size=1)
    for i, item in enumerate(loader_adience_race):
        print('i', i)
        x, u, a, s = item
        print(a)
        break




