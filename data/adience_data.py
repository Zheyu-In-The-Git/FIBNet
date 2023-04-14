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

import mat73


class AdienceData(data.Dataset):
    def __init__(self,
                 dim_img:int,
                 data_dir:str,
                 identity_nums:int,
                 sensitive_attr:str,
                 ):

        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_attr = sensitive_attr
        self.identity_nums = identity_nums

        fn = partial(os.path.join, self.data_dir)
        adience_dataset_fold_0 = pandas.read_csv(fn('fold_0_data.txt'), index_col=False)
        adience_dataset_fold_1 = pandas.read_csv(fn('fold_1_data.txt'), index_col=False)
        adience_dataset_fold_2 = pandas.read_csv(fn('fold_2_data.txt'), index_col=False)
        adience_dataset_fold_3 = pandas.read_csv(fn('fold_3_data.txt'), index_col=False)
        adience_dataset_fold_4 = pandas.read_csv(fn('fold_4_data.txt'), index_col=False)

        self.adience_dateset = pd.concat([adience_dataset_fold_0, adience_dataset_fold_1,
                                          adience_dataset_fold_2, adience_dataset_fold_3,
                                          adience_dataset_fold_4])

        self.trans = transforms.Compose([
            transforms.Resize(self.dim_img),
            transforms.CenterCrop((178, 178)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


    def __len__(self):
        return self.adience_dateset.shape[0]

    def __getitem__(self, index):




        img_path = os.path.join(self.data_dir, 'faces')









