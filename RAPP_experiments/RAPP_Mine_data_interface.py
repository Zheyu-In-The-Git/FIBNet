
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl

from RAPP_Mine_data import CelebaRAPPMineGenderData, CelebaRAPPMineRaceData, LFWRAPPMineGenderData, LFWRAPPMineRaceData, AdienceMineGenderData, AdienceMineRaceData

import platform
import sys


######################################################################
###################### CelebA  #######################################
######################################################################


class CelebaRAPPMineTrainingDatasetInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers : int,
                 dataset_name: str,
                 batch_size : int,
                 dim_img : int,
                 data_dir : str,
                 identity_nums : int,
                 sensitive_attr:str,
                 **kargs):
        super(CelebaRAPPMineTrainingDatasetInterface).__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums
        self.sensitive_attr = sensitive_attr


        self.pin_memory = kargs['pin_memory']
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = True


        if self.sensitive_attr == 'Male':
            self.training_dataset = CelebaRAPPMineGenderData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='train_63%')
            self.test_dataset = CelebaRAPPMineGenderData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='test_30%')
        elif self.sensitive_attr == 'Race':
            self.training_dataset = CelebaRAPPMineRaceData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='train_63%')
            self.test_dataset = CelebaRAPPMineRaceData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='test_30%')
        else:
            print('the dataset has not define')
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.Train_Dataset = self.training_dataset
        if stage == 'test' or stage is None:
            self.Test_Dataset = self.test_dataset
    def train_dataloader(self):
        # sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset) * 20)
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)
    def test_dataloader(self):
        return DataLoader(self.Test_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)



class CelebaRAPPMineTestDatasetInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers: int,
                 dataset_name: str,
                 batch_size: int,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int,
                 sensitive_attr: str,
                 **kargs):
        super(CelebaRAPPMineTestDatasetInterface).__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums
        self.sensitive_attr = sensitive_attr

        self.pin_memory = kargs['pin_memory']
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = True

        print('warninig: this dataset has no test dataloader')

        if self.sensitive_attr == 'Male':
            self.training_dataset = CelebaRAPPMineGenderData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='test_30%')
        elif self.sensitive_attr == 'Race':
            self.training_dataset = CelebaRAPPMineRaceData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=10177, split='test_30%')
        else:
            print('the dataset has not define')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.Train_Dataset = self.training_dataset

    def train_dataloader(self):
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)


###################################################################
###################### LFW  #######################################
###################################################################

class LFWRAPPMineDatasetInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers: int,
                 dataset_name: str,
                 batch_size: int,
                 dim_img: int,
                 data_dir: str,
                 identity_nums: int,
                 sensitive_attr: str,
                 **kargs):
        super(LFWRAPPMineDatasetInterface).__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.identity_nums = identity_nums
        self.sensitive_attr = sensitive_attr

        self.pin_memory = kargs['pin_memory']
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = True

        print('warninig: this dataset has no test dataloader')

        self.prepare_data_per_node = True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = True



        if sys.platform.startswith('darwin'): # 判断当前是否是mac系统
            self.LFWData_img_path_replace = True  # mac 处理lfw 把图像路径\\ -> /
            self.LFWRecognition_img_path_replace = False  # mac 不需要处理
        else:
            self.LFWData_img_path_replace = False  # windows 不需要处理 直接用\\
            self.LFWRecognition_img_path_replace = True  # windows 要把图像路径/ -> \\




        if self.sensitive_attr == 'Male':
            self.training_dataset = LFWRAPPMineGenderData(dim_img=self.dim_img, data_dir=self.data_dir, split='all', sensitive_attr='Male', img_path_replace=self.LFWData_img_path_replace, identity_nums=2088)
        elif self.sensitive_attr == 'Race':
            self.training_dataset = LFWRAPPMineRaceData(dim_img=self.dim_img, data_dir=self.data_dir, split='all', sensitive_attr='White', img_path_replace=self.LFWData_img_path_replace, identity_nums=2088)
        else:
            print('the dataset has not define')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.Train_Dataset = self.training_dataset

    def train_dataloader(self):
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

###################################################################
###################### Adience  #######################################
###################################################################


class AdienceRAPPMineDatasetInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers: int,
                 dataset_name: str,
                 batch_size: int,
                 dim_img: int,
                 identity_nums: int,
                 sensitive_attr: str,
                 data_dir: str,
                 **kwargs):
        super(AdienceRAPPMineDatasetInterface).__init__()
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.data_dir = data_dir

        # 数据集相关参数
        self.dim_img = dim_img
        self.identity = identity_nums
        self.sensitive_attr = sensitive_attr
        self.pin_memory = kwargs['pin_memory']

        self.prepare_data_per_node = True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = True

        if self.sensitive_attr == 'Male':
            self.training_dataset = AdienceMineGenderData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=2284)
        elif self.sensitive_attr == 'Race':
            self.training_dataset = AdienceMineRaceData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=2284)
        else:
            print('the dataset has not defined')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.Train_Dataset = self.training_dataset

    def train_dataloader(self):
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)








if __name__ == '__main__':
    # 测试用例
    celeba_data_dir = '/Volumes/xiaozhe_SSD/datasets/celeba'
    lfw_data_dir = '/Volumes/xiaozhe_SSD/datasets/lfw/lfw112'
    adience_data_dir = '/Volumes/xiaozhe_SSD/datasets/Adience'

    # gender
    celeba_training_gender_dataloader = CelebaRAPPMineTrainingDatasetInterface(num_workers=0, dataset_name='CelebA_training_dataset', batch_size=1, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False)
    celeba_test_gender_dataloader = CelebaRAPPMineTestDatasetInterface(num_workers=0, dataset_name='CelebA_test_dataset', batch_size=1, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False)
    lfw_gender_dataloader = LFWRAPPMineDatasetInterface(num_workers=0, dataset_name='LFW', batch_size=1, dim_img=224, data_dir=lfw_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False)
    adience_gender_dataloader = AdienceRAPPMineDatasetInterface(num_workers=0, dataset_name='Adience', batch_size=1, dim_img=224, data_dir=adience_data_dir, identity_nums=10177, sensitive_attr='Male', pin_memory=False)

    # race
    celeba_training_race_dataloader = CelebaRAPPMineTrainingDatasetInterface(num_workers=0, dataset_name='CelebA_training_dataset', batch_size=1, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False)
    celeba_test_race_dataloader = CelebaRAPPMineTestDatasetInterface(num_workers=0, dataset_name='CelebA_test_dataset', batch_size=1, dim_img=224, data_dir=celeba_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False)
    lfw_race_dataloader = LFWRAPPMineDatasetInterface(num_workers=0, dataset_name='LFW', batch_size=1, dim_img=224, data_dir=lfw_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False)
    adience_race_dataloader = AdienceRAPPMineDatasetInterface(num_workers=0, dataset_name='Adience', batch_size=1, dim_img=224, data_dir=adience_data_dir, identity_nums=10177, sensitive_attr='Race', pin_memory=False)


    
    for dataloader in [celeba_training_gender_dataloader, celeba_test_gender_dataloader, lfw_gender_dataloader, adience_gender_dataloader]:
        dataloader.setup(stage='fit')

        for i, item in enumerate(dataloader.train_dataloader()):
            print('dataloader:', dataloader.dataset_name)
            x, u, a, s = item
            print(a)
            break


    for dataloader in [celeba_training_race_dataloader, celeba_test_race_dataloader, lfw_race_dataloader, adience_race_dataloader]:
        dataloader.setup(stage='fit')

        for i, item in enumerate(dataloader.train_dataloader()):
            print('dataloader:', dataloader.dataset_name)
            x, u, a, s = item
            print(a)
            break

    '''
    
    def dataloader_test(dataloader):
        dataloader.setup(stage='fit')
        for i, item in enumerate(dataloader.train_dataloader()):
            print('dataloader:', dataloader.dataset_name)
            x, u, a, s = item
            print(a)
            break

    dataloader_test(celeba_training_gender_dataloader)
    '''











