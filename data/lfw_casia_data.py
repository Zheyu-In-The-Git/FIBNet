import torch
import os
import numpy as np
import pandas as pd
import PIL
import torch.utils.data as data
from functools import partial
import re

from torchvision import transforms

from torch.utils.data import DataLoader

import mat73

class LFWCasiaData(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 lfw_data_dir:str,
                 casia_data_dir:str,
                 sensitive_attr: str,
                 img_path_replace: bool,
                 split: str
                 ):
        '''
        暂且不实现身份的遍历，但仍然会返回变量u，u一直返回0
        '''
        self.dim_img = dim_img
        self.lfw_data_dir = lfw_data_dir
        self.casia_data_dir = casia_data_dir
        self.sensitive_attr = sensitive_attr
        self.img_path_replace = img_path_replace
        identity_nums = 5749


        ###########################################
        ###############LFW部分######################
        ##########################################

        lfw_path_fn = partial(os.path.join, self.lfw_data_dir)
        self.lfw_dataset = pd.read_csv(lfw_path_fn('lfw_att_73.csv'))
        lfw_dataset_load_indices_train_test = mat73.loadmat(lfw_path_fn('indices_train_test.mat'))
        self.lfw_dataset_id = pd.read_csv(lfw_path_fn('lfw_train_test_id.csv'), index_col='name')

        if split == 'train':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_train']
        elif split == 'test':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_test']
        elif split == 'all':
            self.lfw_dataset_indices = np.append(lfw_dataset_load_indices_train_test['indices_img_train'],lfw_dataset_load_indices_train_test['indices_img_test'])
        else:
            'please input the correct lfw dataset split string'

        self.lfw_dataset_img_path = self.lfw_dataset.iloc[:, 0]  # 路径这里需要重新思考

        self.lfw_sensitive_data = self.lfw_dataset[self.sensitive_attr]


        lfw_img_local_device_path =[]
        for i in self.lfw_dataset_img_path:
            i = lfw_path_fn('img',i)
            lfw_img_local_device_path.append(i)
            #print(i)


        lfw_data_imgpath_sensitiveattr = np.column_stack((lfw_img_local_device_path, self.lfw_sensitive_data))
        #print(lfw_data_imgpath_sensitiveattr)



        ##########################################
        ###############Casia部分###################
        ##########################################

        casia_path_fn = partial(os.path.join, self.casia_data_dir)

        casia_data_imgpath_sensitiveattr = []
        if self.sensitive_attr == 'Male':
            casia_id_subimg_gender_csv_path = casia_path_fn('csv_file', 'CasiaFace_id_subpath_gender.csv')
            casia_id_subimg_gender_data = pd.read_csv(casia_id_subimg_gender_csv_path)
            #print(casia_id_subimg_gender_data)
            casia_id_subimg_gender_data['identity'] = casia_id_subimg_gender_data['identity'].astype(str).apply(lambda x: re.sub(r'^(\d)$', r'00\1', x))
            casia_id_subimg_gender_data['identity'] = casia_id_subimg_gender_data['identity'].astype(str).apply(lambda x: re.sub(r'^(\d{2})$', r'0\1', x))


            casia_img_path_list = []
            for i in range(casia_id_subimg_gender_data.shape[0]):
                casia_img_path = casia_path_fn('dataset',casia_id_subimg_gender_data.loc[i]['identity'], casia_id_subimg_gender_data.loc[i]['sub_img_path'])
                casia_img_path = casia_img_path.replace('.bmp', '.jpg')
                casia_img_path_list.append(casia_img_path)

            casia_id_subimg_gender_data['gender'] = casia_id_subimg_gender_data['gender'].replace(1, '1.0')
            casia_id_subimg_gender_data['gender'] = casia_id_subimg_gender_data['gender'].replace(0, '0.0')

            casia_data_imgpath_sensitiveattr = np.column_stack((casia_img_path_list, casia_id_subimg_gender_data['gender']))



        elif self.sensitive_attr == 'White':
            casia_id_subimg_gender_csv_path = casia_path_fn('csv_file', 'CasiaFace_id_subpath_gender.csv')
            casia_id_subimg_gender_data = pd.read_csv(casia_id_subimg_gender_csv_path)
            casia_id_subimg_gender_data['identity'] = casia_id_subimg_gender_data['identity'].astype(str).apply(
                lambda x: re.sub(r'^(\d)$', r'00\1', x))
            casia_id_subimg_gender_data['identity'] = casia_id_subimg_gender_data['identity'].astype(str).apply(
                lambda x: re.sub(r'^(\d{2})$', r'0\1', x))

            casia_img_path_list = []
            for i in range(casia_id_subimg_gender_data.shape[0]):
                casia_img_path = casia_path_fn('dataset', casia_id_subimg_gender_data.loc[i]['identity'],
                                               casia_id_subimg_gender_data.loc[i]['sub_img_path'])
                casia_img_path = casia_img_path.replace('.bmp', '.jpg')
                casia_img_path_list.append(casia_img_path)

            asian_race = np.zeros(2500)

            casia_data_imgpath_sensitiveattr = np.column_stack((casia_img_path_list, asian_race))
            #print(casia_data_imgpath_sensitiveattr)



        ##########################################
        ###############LFW + Casia部分############
        ##########################################

        lfw_casia_database_np = np.row_stack((lfw_data_imgpath_sensitiveattr, casia_data_imgpath_sensitiveattr))
        self.lfw_casia_database_pd = pd.DataFrame(lfw_casia_database_np, columns=['img_path', 'sensitive_attr'])
        #print(self.lfw_casia_database_pd.shape)

        ######################################
        ###############数据预处理部分############
        ######################################
        self.trans = transforms.Compose([
            transforms.Resize((self.dim_img, self.dim_img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return self.lfw_casia_database_pd.shape[0]

    def __getitem__(self, index):


        x = PIL.Image.open(self.lfw_casia_database_pd.loc[index]['img_path'])
        x = self.trans(x)

        u = 0
        u = torch.tensor([u]).to(torch.float32)


        s = self.lfw_casia_database_pd.loc[index]['sensitive_attr']
        s = float(s)
        s = torch.tensor([s]).to(torch.float32)
        return x, u, s








if __name__ == '__main__':
    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'


    loader = LFWCasiaData(dim_img=224,
                          lfw_data_dir=lfw_data_dir,
                          casia_data_dir=casia_data_dir,
                          sensitive_attr='Male', # Male or White
                          img_path_replace=True,
                          split='all')
    train_loader = DataLoader(loader, batch_size=200, shuffle=True)
    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x.shape)
        print(u)
        print(s)






    '''
    train_loader = DataLoader(loader, batch_size=3, shuffle=False)

    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x.shape)
        print(u)
        print(s.shape)
        break
    '''















