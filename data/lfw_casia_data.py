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





class LFWCaisaRecognitionTestPairs(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 lfw_data_dir:str,
                 casia_data_dir:str):
        self.dim_img = dim_img
        self.lfw_data_dir = lfw_data_dir
        self.casia_data_dir = casia_data_dir


        ##############################
        ###########LFW################
        ##############################
        lfw_fn = partial(os.path.join, self.lfw_data_dir)
        lfw_test_dataset = pd.read_csv(lfw_fn('lfw_test_pair.txt'), delim_whitespace=True, header=None, index_col=None)
        lfw_test_dataset.columns = ['img_x', 'img_y', 'match']
        #print(lfw_test_dataset)

        lfw_img_x_local_device_path = []
        for i in lfw_test_dataset['img_x']:
            i = lfw_fn('img', i)
            i = i.replace('/', '\\')
            lfw_img_x_local_device_path.append(i)


        lfw_img_y_local_device_path = []
        for i in lfw_test_dataset['img_y']:
            i = lfw_fn('img', i)
            i = i.replace('/','\\')
            lfw_img_y_local_device_path.append(i)


        lfw_data_img_xy = np.column_stack((lfw_img_x_local_device_path, lfw_img_y_local_device_path))


        lfw_data_img_xy_match = np.column_stack((lfw_data_img_xy, lfw_test_dataset['match']))


        #lfw最终的数据 两个图像数据对应的匹配值
        lfw_face_recognition_data = lfw_data_img_xy_match
        #print(lfw_face_recognition_data)






        ##############################
        ###########Casia##############
        ##############################
        casia_fn = partial(os.path.join, self.casia_data_dir)
        casia_test_dataset = pd.read_csv(casia_fn('csv_file', 'casia_face_verify_test_dataset.csv'))
        casia_test_dataset = casia_test_dataset.drop(casia_test_dataset.columns[0],axis=1)
        #print(casia_test_dataset)

        casia_img_x_list = []
        for i in casia_test_dataset['img_x']:
            i = casia_fn('dataset', i[0:3], i)
            i = i.replace('bmp', 'jpg')
            casia_img_x_list.append(i)


        casia_img_y_list = []
        for i in casia_test_dataset['img_y']:
            i = casia_fn('dataset', i[0:3], i)
            i = i.replace('bmp', 'jpg')
            casia_img_y_list.append(i)

        casia_data_img_xy = np.column_stack((casia_img_x_list, casia_img_y_list))
        casia_data_img_xy_match = np.column_stack((casia_data_img_xy, casia_test_dataset['match']))
        #print(casia_data_img_xy_match)



        ###############################
        ########## LfW+Casia###########
        ###############################
        lfw_casia_database_np = np.row_stack((lfw_face_recognition_data, casia_data_img_xy_match))
        self.lfw_casia_database_pd = pd.DataFrame(lfw_casia_database_np, columns=['img_x','img_y', 'match'])
        #print(self.lfw_casia_database_pd)


        self.trans = transforms.Compose([transforms.Resize((self.dim_img, self.dim_img)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5,0.5, 0.5], std=[0.5,0.5,0.5])])
    def __len__(self):
        return self.lfw_casia_database_pd.shape[0]


    def __getitem__(self, index):

        img_x_path = self.lfw_casia_database_pd['img_x'][index]
        img_y_path = self.lfw_casia_database_pd['img_y'][index]

        img_x = PIL.Image.open(img_x_path)
        img_x = self.trans(img_x)

        img_y = PIL.Image.open(img_y_path)
        img_y = self.trans(img_y)

        match = torch.tensor(float(self.lfw_casia_database_pd['match'][index]))

        return img_x, img_y, match







if __name__ == '__main__':

    ###################################
    ############# 人脸属性的dataloader###
    ###################################
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
        #print('i', i)
        x, u, s = item
        #print(x.shape)
        #print(u)
        #print(s)
        break

    face_loader_data = LFWCaisaRecognitionTestPairs(dim_img=224,
                                               lfw_data_dir=lfw_data_dir,
                                               casia_data_dir=casia_data_dir)
    face_loader = DataLoader(face_loader_data, batch_size=100, shuffle=True)
    for i, item in enumerate(face_loader):
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)






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















