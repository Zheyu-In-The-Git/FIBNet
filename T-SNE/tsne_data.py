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
import mat73
from torchvision import transforms
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data import DataLoader
import os
import re

from sklearn.utils import shuffle


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

        self.trans = transforms.Compose([transforms.CenterCrop(150),
                                         transforms.Resize(self.dim_img),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset.index.values[index]
        X = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba', img_path))

        x = self.trans(X)


        u = self.id.loc[img_path].values
        u = torch.as_tensor(u) - 1.0
        u = u.long()

        s = torch.as_tensor(self.dataset.values[index])
        s = torch.div(s + 1, 2, rounding_mode='floor')
        s = s.to(torch.float32)
        return x, u, s


class CelebaTSNERaceExperiment(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 split:str):
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.split = split

        split_map = {
            'train':0,
            'valid':1,
            'test':2,
            'all':None,
            'train_63%':'train_63%',
            'valid_7%':'valid_7%',
            'test_30%':'test_30%'
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", 'train_valid_70%', 'test_30%'))]

        fn = partial(os.path.join, self.data_dir)
        imgpath_id = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names=['img_path', 'id'])

        imgpath_white = pd.read_csv(fn('celeba_imgpath_race.csv'))
        imgpath_white = imgpath_white.drop(imgpath_white.columns[0], axis=1)
        imgpath_race_id = pd.merge(imgpath_white, imgpath_id, on='img_path', how='left')

        print(imgpath_race_id)

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
            self.trans = transforms.Compose([transforms.CenterCrop(150),
                                             transforms.Resize(self.dim_img),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.dataset = imgpath_race_id[mask]


        #print(white_select_condition)
        white_dataset = self.dataset.where(self.dataset['white'] == 1.0)
        white_dataset = white_dataset.dropna(axis=0)
        white_dataset = white_dataset.head(500)
        #print(white_dataset)

        colored_people = self.dataset.where(self.dataset['white'] == 0.0)
        colored_people = colored_people.dropna(axis=0)
        colored_people = colored_people.head(500)
        #print(colored_people)

        dataset_samples = pd.concat([white_dataset, colored_people])
        dataset_samples = dataset_samples.reset_index()
        self.dataset_samples = dataset_samples.drop('index', axis=1)
        self.dataset_samples = self.dataset_samples.values
        print(self.dataset_samples)



        self.trans = transforms.Compose([
            transforms.Resize(self.dim_img),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):

        X = PIL.Image.open(os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba', self.dataset_samples[index][0]))

        x = self.trans(X)

        u = self.dataset_samples[index][2]
        u = torch.as_tensor(u) - 1.0
        u = u.long()

        s = torch.as_tensor(self.dataset_samples[index][1])
        s = torch.tensor([s]).to(torch.float32)

        return x, u, s




class LFWTSNEExperiment(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 sensitive_attr:str,
                 img_path_replace:bool,
                 split:str):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr

        fn = partial(os.path.join, self.data_dir)
        self.lfw_dataset = pandas.read_csv(fn('lfw_att_73.csv'))


        imgpath_white_male = self.lfw_dataset.where(self.lfw_dataset[self.sensitive_attr] == 1.0) # 男性的标签为1，白人的标签为1，而不是既是白人也是男性的意思
        imgpath_white_male = imgpath_white_male.dropna(axis=0)
        imgpath_white_male = imgpath_white_male.head(500)
        #print(imgpath_white_male)

        imgpath_colored_race_female = self.lfw_dataset.where(self.lfw_dataset[self.sensitive_attr] == 0.0) # 女性的标签为0，有色人种的标签为0，而不是既是女性也是有色人种的意思
        imgpath_colored_race_female = imgpath_colored_race_female.dropna(axis=0)
        imgpath_colored_race_female = imgpath_colored_race_female.head(500)
        #print(imgpath_colored_race_female)

        self.lfw_dataset = pd.concat([imgpath_white_male, imgpath_colored_race_female])
        self.lfw_dataset = self.lfw_dataset.reset_index()
        self.lfw_dataset = self.lfw_dataset.drop('index', axis=1)


        lfw_dataset_load_indices_train_test = mat73.loadmat(fn('indices_train_test.mat'))
        self.lfw_dataset_id = pandas.read_csv(fn('lfw_train_test_id.csv'), index_col='name')

        if split == 'train':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_train']
        elif split == 'test':
            self.lfw_dataset_indices = lfw_dataset_load_indices_train_test['indices_img_test']
        elif split == 'all':
            self.lfw_dataset_indices = np.append(lfw_dataset_load_indices_train_test['indices_img_train'],
                                                 lfw_dataset_load_indices_train_test['indices_img_test'])
        else:
            'please input the correct lfw dataset split string'



        self.lfw_dataset_img_path = self.lfw_dataset.iloc[:, 0]  # 路径这里需要重新思考


        self.trans = transforms.Compose([
            transforms.Resize(self.dim_img),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.img_path_replace = img_path_replace

    def __len__(self):
        return len(self.lfw_dataset_img_path)

    def __getitem__(self, index):

        img_path_name = self.lfw_dataset_img_path[index]

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


        s = self.lfw_dataset.loc[index, self.sensitive_attr]
        s = torch.tensor([s]).to(torch.float32)
        return x, u, s



class AdienceTSNEGenderExperiment(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 sensitive_attr:str):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr

        fn = partial(os.path.join, self.data_dir)
        adience_dataset_fold_0 = pandas.read_table(fn('fold_0_data.txt'), index_col=False)
        adience_dataset_fold_1 = pandas.read_table(fn('fold_1_data.txt'), index_col=False)
        adience_dataset_fold_2 = pandas.read_table(fn('fold_2_data.txt'), index_col=False)
        adience_dataset_fold_3 = pandas.read_table(fn('fold_3_data.txt'), index_col=False)
        adience_dataset_fold_4 = pandas.read_table(fn('fold_4_data.txt'), index_col=False)

        adience_dataset = pd.concat([adience_dataset_fold_0, adience_dataset_fold_1,
                                     adience_dataset_fold_2, adience_dataset_fold_3,
                                     adience_dataset_fold_4], ignore_index=True)

        adience_dataset = adience_dataset.dropna(subset=['gender'])
        adience_dataset = adience_dataset.reset_index()

        self.adience_dataset = adience_dataset[['user_id', 'original_image', 'face_id', 'gender']]

        male_samples = self.adience_dataset.where(self.adience_dataset['gender'] == 'm')
        male_samples = male_samples.dropna(axis=0)
        male_samples = male_samples.head(500)
        #print(male_samples)

        female_samples = self.adience_dataset.where(self.adience_dataset['gender'] == 'f')
        female_samples = female_samples.dropna(axis=0)
        female_samples = female_samples.head(500)
        #print(female_samples)

        self.adience_dataset = pd.concat([male_samples, female_samples])
        self.adience_dataset = self.adience_dataset.reset_index()
        self.adience_dataset = self.adience_dataset.drop('index', axis=1)
        print(self.adience_dataset)



        self.trans_first = transforms.Compose([transforms.CenterCrop((1250, 1250))])
        self.trans_second = transforms.Compose([transforms.Resize((self.dim_img, self.dim_img)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                     std=[0.5, 0.5, 0.5]),
                                                ])

        self.trans = transforms.Compose([transforms.CenterCrop((250, 250)),
                                         transforms.Resize((self.dim_img, self.dim_img)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])



    def __len__(self):
        return self.adience_dataset.shape[0]

    def __getitem__(self, index):

        user_id_path = self.adience_dataset['user_id'][index]
        original_image = self.adience_dataset['original_image'][index]
        face_id = int(self.adience_dataset['face_id'][index])
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
        else:
            s = torch.tensor([1.0])


        return x, u, s


class AdienceTSNERaceExperiment(data.Dataset):
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
        #print(self.userid_imgsubpath_white_faceid['white'])

        white_samples = self.userid_imgsubpath_white_faceid.where(self.userid_imgsubpath_white_faceid['white'] == 1.0)
        white_samples = white_samples.dropna(axis=0)
        white_samples = white_samples.head(500)
        #print(white_samples)

        colored_race_samples = self.userid_imgsubpath_white_faceid.where(self.userid_imgsubpath_white_faceid['white'] == 0.0)
        colored_race_samples = colored_race_samples.dropna(axis=0)
        colored_race_samples = colored_race_samples.head(500)
        #print(colored_race_samples)


        self.userid_imgsubpath_white_faceid = pd.concat([white_samples, colored_race_samples])
        self.userid_imgsubpath_white_faceid = self.userid_imgsubpath_white_faceid.reset_index()
        self.userid_imgsubpath_white_faceid = self.userid_imgsubpath_white_faceid.drop('index', axis=1)
        print(self.userid_imgsubpath_white_faceid)



        self.trans = transforms.Compose([transforms.CenterCrop((300, 300)),
                                         transforms.Resize((self.dim_img, self.dim_img)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

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

        return x, u, s


class LFWCasiaFaceExperiment(data.Dataset):
    def __init__(self,
                 dim_img: int,
                 lfw_data_dir: str,
                 casia_data_dir: str,
                 sensitive_attr: str,
                 img_path_replace: bool,
                 split: str):

        self.dim_img = dim_img
        self.lfw_data_dir = lfw_data_dir
        self.casia_data_dir = casia_data_dir
        self.sensitive_attr = sensitive_attr
        self.img_path_replace = img_path_replace

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
            self.lfw_dataset_indices = np.append(lfw_dataset_load_indices_train_test['indices_img_train'],
                                                 lfw_dataset_load_indices_train_test['indices_img_test'])
        else:
            'please input the correct lfw dataset split string'

        self.lfw_dataset_img_path = self.lfw_dataset.iloc[:, 0]  # 路径这里需要重新思考

        self.lfw_sensitive_data = self.lfw_dataset[self.sensitive_attr]

        lfw_img_local_device_path = []
        for i in self.lfw_dataset_img_path:
            i = lfw_path_fn('img', i)
            lfw_img_local_device_path.append(i)
            # print(i)

        lfw_data_imgpath_sensitiveattr = np.column_stack((lfw_img_local_device_path, self.lfw_sensitive_data))
        # print(lfw_data_imgpath_sensitiveattr)

        ##########################################
        ###############Casia部分###################
        ##########################################

        casia_path_fn = partial(os.path.join, self.casia_data_dir)

        casia_data_imgpath_sensitiveattr = []
        if self.sensitive_attr == 'Male':
            casia_id_subimg_gender_csv_path = casia_path_fn('csv_file', 'CasiaFace_id_subpath_gender.csv')
            casia_id_subimg_gender_data = pd.read_csv(casia_id_subimg_gender_csv_path)
            # print(casia_id_subimg_gender_data)
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

            casia_id_subimg_gender_data['gender'] = casia_id_subimg_gender_data['gender'].replace(1, '1.0')
            casia_id_subimg_gender_data['gender'] = casia_id_subimg_gender_data['gender'].replace(0, '0.0')

            casia_data_imgpath_sensitiveattr = np.column_stack(
                (casia_img_path_list, casia_id_subimg_gender_data['gender']))



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

        ##########################################
        ###############LFW + Casia部分############
        ##########################################

        lfw_casia_database_np = np.row_stack((lfw_data_imgpath_sensitiveattr, casia_data_imgpath_sensitiveattr))
        lfw_casia_database_pd = pd.DataFrame(lfw_casia_database_np, columns=['img_path', 'sensitive_attr'])


        lfw_casia_database_shuffled_pd = lfw_casia_database_pd.sample(frac=1, random_state=83).reset_index(drop=True)


        lfw_casia_positive_attr_sample = lfw_casia_database_shuffled_pd.where(lfw_casia_database_shuffled_pd['sensitive_attr'] == '1.0') # 对应于男性 和 高加索人
        lfw_casia_positive_attr_sample = lfw_casia_positive_attr_sample.dropna()
        lfw_casia_positive_attr_sample = lfw_casia_positive_attr_sample.head(500)
        #print(lfw_casia_positive_attr_sample)

        lfw_casia_negative_attr = lfw_casia_database_shuffled_pd.where(lfw_casia_database_shuffled_pd['sensitive_attr'] == '0.0') # 对应于女性 和 非高加索人
        lfw_casia_negative_attr = lfw_casia_negative_attr.dropna()
        lfw_casia_negative_attr = lfw_casia_negative_attr.head(500)
        #print(lfw_casia_negative_attr)



        lfw_casia_tsne_database = pd.concat([lfw_casia_positive_attr_sample, lfw_casia_negative_attr])
        self.lfw_casia_tsne_database = lfw_casia_tsne_database.reset_index(drop=True)
        print(self.lfw_casia_tsne_database)

        ######################################
        ###############数据预处理部分############
        ######################################
        self.trans = transforms.Compose([
            transforms.Resize((self.dim_img, self.dim_img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return self.lfw_casia_tsne_database.shape[0]

    def __getitem__(self, index):


        x = PIL.Image.open(self.lfw_casia_tsne_database.loc[index]['img_path'])
        x = self.trans(x)

        u = 1
        u = torch.tensor([u]).to(torch.float32)
        u = u.long()


        s = self.lfw_casia_tsne_database.loc[index]['sensitive_attr']
        s = float(s)
        s = torch.tensor([s]).to(torch.float32)
        return x, u, s





if __name__ == '__main__':
    data_dir = 'D:\datasets\celeba'
    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    loader = LFWCasiaFaceExperiment(dim_img=224,
                                    lfw_data_dir=lfw_data_dir,
                                    casia_data_dir=casia_data_dir,
                                    sensitive_attr='White',  # Male or White
                                    img_path_replace=True,
                                    split='all')

    
    train_loader = DataLoader(loader, batch_size=200, shuffle=True)
    for i, item in enumerate(train_loader):
        # print('i', i)
        x, u, s = item
        print(x.shape)
        print(u)
        print(s)
        break










    '''
    loader = CelebaTSNEExperiment(dim_img=112, data_dir=data_dir, split='test_30%',sensitive_attr='Male')
    train_loader = DataLoader(loader, batch_size=2)
    for i, item in enumerate(train_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break


    
    
    celeba_race_loader = CelebaTSNERaceExperiment(dim_img=112, data_dir = data_dir, split='test_30%')
    celeba_race_test_loader = DataLoader(celeba_race_loader, batch_size=2)
    for i, item in enumerate(celeba_race_test_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break
        
    

    lfw_dataset = LFWTSNEExperiment(dim_img=112,  sensitive_attr='Male', img_path_replace=False, split='all', data_dir = lfw_data_dir)
    lfw_loader = DataLoader(lfw_dataset, batch_size=1)

    
    for i, item in enumerate(lfw_loader):
        print('i', i)
        x, u, s = item
        print(x)
        print(u)
        print(s)
        break

    Adience_data_dir = '/Users/xiaozhe/datasets/Adience'
    Adience_dataset = AdienceTSNEGenderExperiment(dim_img=112, sensitive_attr='Male', data_dir=Adience_data_dir)
    for i, item in enumerate(Adience_dataset):
        print('i', i)
        x, u, s  = item
        print(x)
        print(u)
        print(s)
        break

    Adience_race_dataset = AdienceTSNERaceExperiment(dim_img=112, data_dir=Adience_data_dir, identity_nums=2284)
    for i, item in enumerate(Adience_race_dataset):
        print('i', i)
        x,u,s = item
        print(x)
        print(u)
        print(s)
        break
    '''




