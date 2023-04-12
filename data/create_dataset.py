import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from functools import partial

celeba_ssd_path = '/Volumes/xiaozhe_SSD/datasets/celeba'

fn = partial(os.path.join, celeba_ssd_path) # csv检索用的

# 导入数据集的划分
list_eval_partition_data = pd.read_table(fn('list_eval_partition.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','partition'])

# 身份信息对应的图片信息
identity_CelebA = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','id'])

'''--训练集效果---'''
# 训练集作出img partition id 的效果
train_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 0]
# print(validation_data_img_partition)

train_data_img_partition_identity = pd.merge(train_data_img_partition, identity_CelebA, on='img')
# print(train_data_img_partition_identity)

# 去掉partition 项
train_data_img_identity = train_data_img_partition_identity.drop(labels = 'partition', axis=1)
# print(train_data_img_identity)

# 删除重复出现 id 项目
train_data_img_identity_independence = train_data_img_identity.drop_duplicates('id', keep='first')
print(train_data_img_identity_independence)


'''--验证集效果---'''
# 验证集作出 img partition id的效果
validation_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 1]
# print(validation_data_img_partition)

validation_data_img_partition_identity = pd.merge(validation_data_img_partition, identity_CelebA, on='img')
# print(validation_data_img_partition_identity)

validation_data_img_partition_identity_independent = validation_data_img_partition_identity.drop_duplicates('id', keep='last')
# print(validation_data_img_partition_identity_independent)

validation_data_img_partition_identity_sameimg = pd.merge(validation_data_img_partition_identity, train_data_img_identity_independence, on='id')
# print(validation_data_img_partition_identity_sameimg)


'''--测试集效果---'''
# 测试集作出 img partition id 的效果
test_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 2]
test_data_img_partition_identity = pd.merge(test_data_img_partition, identity_CelebA, on='img')
# print(test_data_img_partition_identity)





