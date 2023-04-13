import os
import time

import numpy as np
import numpy.random
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

numpy.random.seed(83)

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
# print(train_data_img_identity_independence)

identity_CelebA_drop_duplicates = identity_CelebA.drop_duplicates('id', keep = 'first')
# print(identity_CelebA_drop_duplicates)


'''--验证集效果---'''
# 验证集作出 img partition id的效果
validation_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 1]
# print(validation_data_img_partition)

validation_data_img_partition_identity = pd.merge(validation_data_img_partition, identity_CelebA, on='img')
# print(validation_data_img_partition_identity)


# 验证集匹配的部分
validation_data_img_partition_identity_sameimg = pd.merge(validation_data_img_partition_identity, identity_CelebA_drop_duplicates, on='id')
# print(validation_data_img_partition_identity_sameimg)

validation_data_img_partition_identity_sameimg.insert(loc = 4, column='match', value=np.ones(validation_data_img_partition_identity_sameimg.shape[0]))

validation_data_img_partition_identity_sameimg_match = validation_data_img_partition_identity_sameimg.copy()
# print('验证集匹配的部分')
# print(validation_data_img_partition_identity_sameimg_match)


# print(validation_data_img_partition_identity['img'].sample(n=validation_data_img_partition_identity.shape[0]))

# 验证集不匹配的部分

temp = validation_data_img_partition_identity.copy()
random_img = validation_data_img_partition_identity['img'].sample(n=validation_data_img_partition_identity.shape[0]).values
temp.insert(loc = 3, column='img_y', value = random_img)
validation_data_img_partition_identity_differentimg = temp.copy()
# print(validation_Data_img_partition_identity_differentimg)

validation_data_img_partition_identity_differentimg_nonmatch = validation_data_img_partition_identity_differentimg.copy()
validation_data_img_partition_identity_differentimg_nonmatch.insert(loc = 4, column='match', value=np.zeros(validation_data_img_partition_identity_sameimg.shape[0]))
validation_data_img_partition_identity_differentimg_nonmatch.rename(columns={'img':'img_x'}, inplace=True)

# 显示两个匹配和不匹配表结果
# print(validation_data_img_partition_identity_sameimg_match)
# print(validation_data_img_partition_identity_differentimg_nonmatch)

# 合并两个表
celeba_facerecognition_validation_dataset = pd.concat([validation_data_img_partition_identity_sameimg_match, validation_data_img_partition_identity_differentimg_nonmatch])
col_list = ['img_x', 'img_y', 'match', 'id', 'partition']
celeba_facerecognition_validation_dataset = celeba_facerecognition_validation_dataset[col_list]
print(celeba_facerecognition_validation_dataset)
# celeba_facerecognition_validation_dataset.to_json('celeba_facerecognition_validation_dataset.txt', orient='records')
# celeba_facerecognition_validation_dataset.to_csv("celeba_facerecognition_validation_dataset.csv", encoding="utf_8_sig")
# 用csv吧


'''--测试集效果---'''
# 测试集作出 img partition id 的效果
test_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 2]
test_data_img_partition_identity = pd.merge(test_data_img_partition, identity_CelebA, on='img')
# print(test_data_img_partition_identity)





