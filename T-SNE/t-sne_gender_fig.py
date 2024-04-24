import matplotlib.backends.backend_pdf
import torch
import pytorch_lightning as pl
import os
import PIL

pl.seed_everything(83)


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import scienceplots
import os

plt.style.use(['science', 'ieee'])

plt.figure(figsize=(7.16, 8.5))


# 查看一下
# 先做gender的吧
# 用路径吧
'''
###############################################################
################## 制作性别的T-SNE图像 ##########################
###############################################################

'''


def release_datasets(t_sne_datasets_path):

    array_data = np.array([])

    data = np.load(t_sne_datasets_path)
    array_names = data.keys()
    for array_name in array_names:
        array_data = data[array_name]
        
    return array_data

def assignment(t_sne_datasets, boolean_value):
    '''
    将原始数据分配为全1的数据或全0的数据，
    :param t_sne_datasets:
    :param boolean_value:
    :return: 返回X, Y 的坐标
    '''
    datasets_index = np.where(t_sne_datasets[:, 2] == boolean_value)
    return t_sne_datasets[datasets_index]










# arcface相关降维数据
Arcface_celeba_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Arcface_celeba.npz')
Arcface_lfw_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Arcface_lfw_casia.npz')
Arcface_adience_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Arcface_adience.npz')

# bottleneck相关降维数据
Bottleneck_01_celceba_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Bottleneck_0.1_celeba.npz')
Bottleneck_01_lfw_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Bottleneck_0.1_lfw_casia.npz')
Bottleneck_01_adience_path = os.path.abspath(r'files_npz_pdf/t-sne_Male_Bottleneck_0.1_adience.npz')


# 加载Arcface相关数据集
Arcface_celeba_dataset = release_datasets(Arcface_celeba_path)
Arcface_lfw_dataset = release_datasets(Arcface_lfw_path)
Arcface_adience_dataset = release_datasets(Arcface_adience_path)

# 加载Bottleneck相关数据集
Bottleneck_01_celeba_dataset = release_datasets(Bottleneck_01_celceba_path)
Bottleneck_01_lfw_dataset = release_datasets(Bottleneck_01_lfw_path)
Bottleneck_01_adience_dataset = release_datasets(Bottleneck_01_adience_path)



# 男性的数据集 Arcface
Arcface_celeba_boolean_1 = assignment(Arcface_celeba_dataset, 1.0)
Arcface_lfw_boolean_1 = assignment(Arcface_lfw_dataset, 1.0)
Arcface_adience_boolean_1 = assignment(Arcface_adience_dataset, 1.0)


# 男性的数据集 Bottlenck
Bottleneck_01_celeba_boolean_1 = assignment(Bottleneck_01_celeba_dataset, 1.0)
Bottleneck_01_lfw_boolean_1 = assignment(Bottleneck_01_lfw_dataset, 1.0)
Bottleneck_01_adience_boolean_1 =assignment(Bottleneck_01_adience_dataset, 1.0)


# 女性的数据集 Arcface
Arcface_celeba_boolean_0 = assignment(Arcface_celeba_dataset, 0.0)
Arcface_lfw_boolean_0 = assignment(Arcface_lfw_dataset, 0.0)
Arcface_adience_boolean_0 = assignment(Arcface_adience_dataset, 0.0)

# 女性的数据集 Bottleneck
Bottleneck_01_celeba_boolean_0 = assignment(Bottleneck_01_celeba_dataset, 0.0)
Bottleneck_01_lfw_boolean_0 = assignment(Bottleneck_01_lfw_dataset, 0.0)
Bottleneck_01_adience_boolean_0 =assignment(Bottleneck_01_adience_dataset, 0.0)

print(Bottleneck_01_adience_boolean_0[:, 0])





with plt.style.context(['science','ieee', 'high-contrast','grid']):
    fig, ((arcface_celeba_ax, arcface_lfw_ax, arcface_adience_ax),
          (bottleneck_celeba_ax, bottleneck_lfw_ax, bottleneck_adience_ax)) = plt.subplots(2, 3, figsize=(7.16, 4.0))


    # Arcface

    # CelebA数据
    arcface_celeba_ax.scatter(Arcface_celeba_boolean_1[:, 0], Arcface_celeba_boolean_1[:, 1], label='Male', s=5 , c = 'b', marker='.', alpha=0.5)
    arcface_celeba_ax.scatter(Arcface_celeba_boolean_0[:, 0], Arcface_celeba_boolean_0[:, 1], label='Female', s=5, c = 'r', marker='.', alpha=0.5)
    #arcface_celeba_ax.autoscale(tight=True)
    arcface_celeba_ax.set_title('CelebA')
    arcface_celeba_ax.set(ylabel='Arcface')


    # LFW 数据
    arcface_lfw_ax.scatter(Arcface_lfw_boolean_1[:, 0], Arcface_lfw_boolean_1[:, 1], label='Male', s=5, c='b', marker='.', alpha=0.5)
    arcface_lfw_ax.scatter(Arcface_lfw_boolean_0[:, 0], Arcface_lfw_boolean_0[:, 1], label='Female', s=5, c='r', marker='.', alpha=0.5)
    #arcface_lfw_ax.autoscale(tight=True)
    arcface_lfw_ax.set_title('LFW+CASIA-FaceV5')

    # Adience 数据
    arcface_adience_ax.scatter(Arcface_adience_boolean_1[:, 0], Arcface_adience_boolean_1[:, 1], label='Male', s=5, marker='.', c='b', alpha=0.5)
    arcface_adience_ax.scatter(Arcface_adience_boolean_0[:, 0], Arcface_adience_boolean_0[:, 1], label='Female', s=5, marker='.', c='r', alpha=0.5)
    #arcface_adience_ax.autoscale(tight=True)
    arcface_adience_ax.set_title('Adience')


    # Bottlenck
    bottleneck_celeba_ax.scatter(Bottleneck_01_celeba_boolean_1[:, 0], Bottleneck_01_celeba_boolean_1[:, 1], label='Male', s=5, c='b', marker='.', alpha=0.5)
    bottleneck_celeba_ax.scatter(Bottleneck_01_celeba_boolean_0[:, 0], Bottleneck_01_celeba_boolean_0[:, 1], label='Female', s=5, c='r', marker='.', alpha=0.5)
    bottleneck_celeba_ax.set(ylabel=r'$\beta$ = 0.1')


    bottleneck_lfw_ax.scatter(Bottleneck_01_lfw_boolean_1[:, 0], Bottleneck_01_lfw_boolean_1[:, 1], label='Male', s=5, c='b', marker='.', alpha=0.5)
    bottleneck_lfw_ax.scatter(Bottleneck_01_lfw_boolean_0[:, 0], Bottleneck_01_lfw_boolean_0[:, 1], label='Female', s=5, c='r', marker='.', alpha=0.5)


    obj_male = bottleneck_adience_ax.scatter(Bottleneck_01_adience_boolean_1[:, 0], Bottleneck_01_adience_boolean_1[:, 1], label='Male', s=5, c='b', marker='.', alpha=0.5)
    obj_female = bottleneck_adience_ax.scatter(Bottleneck_01_adience_boolean_0[:, 0], Bottleneck_01_adience_boolean_0[:, 1], label='Female', s=5, c='r', marker='.', alpha=0.5)

    fig.legend((obj_male, obj_female),['Male', 'Female'],
               loc="lower center", ncol=6, bbox_to_anchor=(0, 0.0, 1, 0))  # bbox_to_anchor=(0, 1.02, 1, 0.1)


    fig.savefig('t-sne-gender.pdf', dpi=1000)










