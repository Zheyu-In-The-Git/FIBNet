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

from .RAPP_Mine_data import CelebaRAPPMineGenderData, CelebaRAPPMineRaceData, LFWRAPPMineGenderData, LFWRAPPMineRaceData, AdienceMineGenderData, AdienceMineRaceData





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







if __name__ == '__main__':
    # 测试下午做吧
    # celeba_data_dir =
    pass






