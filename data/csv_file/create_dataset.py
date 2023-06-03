import os
import time

import numpy as np
import numpy.random
import pandas as pd

from functools import partial

from scipy.io import loadmat

import h5py
import mat73

numpy.random.seed(83)
'''

celeba_ssd_path = '/Volumes/xiaozhe_SSD/datasets/celeba'

fn = partial(os.path.join, celeba_ssd_path) # csv检索用的

# 导入数据集的划分
list_eval_partition_data = pd.read_table(fn('list_eval_partition.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','partition'])

# 身份信息对应的图片信息
identity_CelebA = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','id'])

# id 是img_x的id

# --训练集效果---
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
'''

'''
# 验证集效果
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
# print(celeba_facerecognition_validation_dataset)
# celeba_facerecognition_validation_dataset.to_json('celeba_facerecognition_validation_dataset.txt', orient='records')
# celeba_facerecognition_validation_dataset.to_csv("celeba_facerecognition_test_dataset.csv", encoding="utf_8_sig")
# 用csv吧
'''

'''


identity_CelebA_drop_duplicates_fortest = identity_CelebA.drop_duplicates('id', keep='last')

# --测试集效果---
# 测试集作出 img partition id 的效果
test_data_img_partition = list_eval_partition_data.loc[list_eval_partition_data['partition'] == 2]
test_data_img_partition_identity = pd.merge(test_data_img_partition, identity_CelebA, on='img')
# print(test_data_img_partition_identity)


# 测试集匹配效果
test_data_img_partition_identity_sameimg = pd.merge(test_data_img_partition_identity, identity_CelebA_drop_duplicates_fortest, on='id')
# print(test_data_img_partition_identity_sameimg)

test_data_img_partition_identity_sameimg.insert(loc = 4, column='match', value=np.ones(test_data_img_partition_identity_sameimg.shape[0]))

test_data_img_partition_identity_sameimg_match = test_data_img_partition_identity_sameimg.copy()
# print('测试集匹配的部分')
# print(test_data_img_partition_identity_sameimg_match)


# 测试集非匹配效果
temp = test_data_img_partition_identity.copy()
random_img = test_data_img_partition_identity['img'].sample(n=test_data_img_partition_identity.shape[0]).values
temp.insert(loc = 3, column='img_y', value = random_img)
test_data_img_partition_identity_differentimg = temp.copy()
# print(test_data_img_partition_identity_differentimg)

test_data_img_partition_identity_differentimg_nonmatch = test_data_img_partition_identity_differentimg.copy()
test_data_img_partition_identity_differentimg_nonmatch.insert(loc = 4, column='match', value=np.zeros(test_data_img_partition_identity_sameimg.shape[0]))
test_data_img_partition_identity_differentimg_nonmatch.rename(columns={'img':'img_x'}, inplace=True)
# print(test_data_img_partition_identity_differentimg_nonmatch)

# 合并两个表

celeba_facerecognition_test_dataset = pd.concat([test_data_img_partition_identity_sameimg_match, test_data_img_partition_identity_differentimg_nonmatch])
col_list = ['img_x', 'img_y', 'match', 'id', 'partition']
celeba_facerecognition_test_dataset = celeba_facerecognition_test_dataset[col_list]
print(celeba_facerecognition_test_dataset)
# celeba_facerecognition_test_dataset.to_csv("celeba_facerecognition_test_dataset.csv", encoding="utf_8_sig")
# 用csv吧

'''


lfw_ssd_path = '/Users/xiaozhe/datasets/lfw/lfw112'
fn_lfw = partial(os.path.join, lfw_ssd_path)



lfw_dataset_load_att_mat = mat73.loadmat(fn_lfw('lfw_att_73.mat'))

print(lfw_dataset_load_att_mat.keys())

lfw_dataset_load_AttName = lfw_dataset_load_att_mat['AttrName']
print(lfw_dataset_load_AttName)
lfw_dataset_load_label = lfw_dataset_load_att_mat['label']
lfw_dataset_load_name = lfw_dataset_load_att_mat['name']

lfw_dataset_pandas = pd.DataFrame(data = lfw_dataset_load_label, index=lfw_dataset_load_name, columns=lfw_dataset_load_AttName)
#print(lfw_dataset_pandas)

# lfw_dataset_pandas.to_csv('lfw_att_40.csv',encoding="utf_8_sig")



'''

lfw_ssd_path = '/Volumes/xiaozhe_SSD/datasets/lfw/lfw112'
fn_lfw = partial(os.path.join, lfw_ssd_path)

lfw_dataset_load_indices_train_test = mat73.loadmat(fn_lfw('indices_train_test.mat'))
print(lfw_dataset_load_indices_train_test)

lfw_dataset_load_indices_train_test_identitytest = lfw_dataset_load_indices_train_test['indices_identity_test']
print(lfw_dataset_load_indices_train_test_identitytest.shape)


lfw_dataset_load_indices_train_test_identitytrain = lfw_dataset_load_indices_train_test['indices_identity_train']
print(lfw_dataset_load_indices_train_test_identitytrain.shape)

lfw_dataset_load_indices_train_test_imgtest = lfw_dataset_load_indices_train_test['indices_img_test']
print(lfw_dataset_load_indices_train_test_imgtest.shape)

lfw_dataset_load_indices_train_test_imgtrain = lfw_dataset_load_indices_train_test['indices_img_train']
print(lfw_dataset_load_indices_train_test_imgtrain.shape)

print(np.where(lfw_dataset_load_indices_train_test_imgtrain == 0))

# 查看lfw的姓名
lfw_dataset_facename = pd.read_table(fn_lfw('lfw-names.txt'),delim_whitespace=True, header=None, index_col=None, names = ['name','number'])
#print(lfw_dataset_facename)

lfw_train_test_indices = np.concatenate((lfw_dataset_load_indices_train_test_identitytrain, lfw_dataset_load_indices_train_test_identitytest))
print(lfw_train_test_indices.shape)

lfw_dataset_facename.insert(2, column='face_id', value=lfw_train_test_indices)
#print(lfw_dataset_facename)

partition_train = np.ones(len(lfw_dataset_load_indices_train_test_identitytrain))
partition_test = np.zeros(len(lfw_dataset_load_indices_train_test_identitytest))
partition = np.concatenate((partition_train, partition_test))
lfw_dataset_facename.insert(3, column='partition', value=partition)
# print(lfw_dataset_facename['face_id'].shape)
# print(lfw_dataset_facename['face_id'].where(lfw_dataset_facename['face_id'] == 0.0).any()) # 是从1开始的
print(lfw_dataset_facename)

# lfw_dataset_facename.to_csv('lfw_train_test_id.csv',encoding="utf_8_sig")

#lfw_attr_data = pd.read_csv(fn_lfw('lfw_att_40.csv'))
#print(lfw_attr_data.shape)

#lfw_dataset_load_indices_train_test_pandas = pd.DataFrame([lfw_dataset_load_indices_train_test])
#print(lfw_dataset_load_indices_train_test_pandas)
'''


'''

# 要做Adience 数据集吗

adience_ssd_path = '/Volumes/xiaozhe_SSD/datasets/Adience'
fn_adience = partial(os.path.join, adience_ssd_path)

adience_dataset_fold_0 = pd.read_table(fn_adience('fold_0_data.txt'),  index_col=False)
adience_dataset_fold_1 = pd.read_table(fn_adience('fold_1_data.txt'),  index_col=False)
adience_dataset_fold_2 = pd.read_table(fn_adience('fold_2_data.txt'),  index_col=False)
adience_dataset_fold_3 = pd.read_table(fn_adience('fold_3_data.txt'),  index_col=False)
adience_dataset_fold_4 = pd.read_table(fn_adience('fold_4_data.txt'),  index_col=False)
# print(adience_dataset_fold_0)
adience_dataset = pd.concat([adience_dataset_fold_0, adience_dataset_fold_1, adience_dataset_fold_2,
                             adience_dataset_fold_3, adience_dataset_fold_4], ignore_index=True)
a = adience_dataset.dropna(subset=['gender'])

# print(a['user_id'][0])
# print(a['original_image'][0])
# print(a['face_id'][0])
# print(a['gender'][0])
# print(a[['user_id', 'original_image', 'face_id', 'gender']])
a = a[['user_id', 'original_image', 'face_id', 'gender']]
a = a.sort_values(by='face_id')
Adience_Dataset = a.reset_index()

Adience_Dataset_first_split = Adience_Dataset.drop_duplicates('face_id', keep='first')
Adience_Dataset_last_split = Adience_Dataset.drop_duplicates('face_id', keep='last')
# print(Adience_Dataset_first_split)
# print(Adience_Dataset_last_split)

adience_merge = pd.merge(Adience_Dataset_first_split, Adience_Dataset_last_split, on='face_id')
adience_merge = adience_merge.drop(labels=['index_x', 'index_y', 'gender_x', 'gender_y'], axis=1)


# print(adience_merge.keys()) # ['user_id_x', 'original_image_x', 'face_id', 'user_id_y', 'original_image_y']

# adience_merge['img_x'] = adience_merge[]

# 图像是匹配的部分

adience_dataset_match_faceverify = pd.DataFrame(columns=['img_x', 'img_y', 'match'])

adience_dataset_match_faceverify['img_x'] = adience_merge['user_id_x'] + '/' + 'coarse_tilt_aligned_face' + '.' + \
                                      adience_merge['face_id'].astype('string') + '.' + adience_merge['original_image_x']
adience_dataset_match_faceverify['img_y'] = adience_merge['user_id_y'] + '/' + 'coarse_tilt_aligned_face' + '.' + \
                                      adience_merge['face_id'].astype('string') + '.' + adience_merge['original_image_y']
adience_dataset_match_faceverify['match'] = np.ones(adience_dataset_match_faceverify.shape[0])




# ______________________________
Adience_Dataset_last_split_random = Adience_Dataset_last_split.take(np.random.permutation(Adience_Dataset_last_split.shape[0]))

temp_first_split = Adience_Dataset_first_split.drop(labels = ['index','gender'], axis=1)
temp_last_split_random = Adience_Dataset_last_split_random.drop(labels = ['index', 'gender'], axis=1)

mapper = {'user_id': 'user_id_x', 'original_image': 'original_image_x', 'face_id':'face_id_x'}
temp_first_split = temp_first_split.rename(mapper, axis='columns')
temp_first_split = temp_first_split.reset_index()
temp_first_split = temp_first_split.drop(labels = ['index'], axis=1)
#print(temp_first_split)


mapper_y = {'user_id': 'user_id_y', 'original_image': 'original_image_y', 'face_id':'face_id_y'}
temp_last_split_random = temp_last_split_random.rename(mapper_y, axis='columns')
temp_last_split_random = temp_last_split_random.reset_index()
temp_last_split_random = temp_last_split_random.drop(labels=['index'], axis=1)
#print(temp_last_split_random)

Adience_Dataset_non_match_merge = pd.concat([temp_first_split, temp_last_split_random], axis=1)
print(Adience_Dataset_non_match_merge['face_id_x'][6])
print(Adience_Dataset_non_match_merge['face_id_y'][6])

adience_dataset_non_match_faceverify = pd.DataFrame(columns=['img_x', 'img_y', 'match'])

adience_dataset_non_match_faceverify['img_x'] = Adience_Dataset_non_match_merge['user_id_x'] + '/' + 'coarse_tilt_aligned_face' + '.' + \
                                      Adience_Dataset_non_match_merge['face_id_x'].astype('string') + '.' + Adience_Dataset_non_match_merge['original_image_x']

adience_dataset_non_match_faceverify['img_y'] = Adience_Dataset_non_match_merge['user_id_y'] + '/' + 'coarse_tilt_aligned_face' + '.' + \
                                      Adience_Dataset_non_match_merge['face_id_y'].astype('string') + '.' + Adience_Dataset_non_match_merge['original_image_y']

adience_dataset_non_match_faceverify['match'] = np.zeros(adience_dataset_non_match_faceverify.shape[0])

print(adience_dataset_non_match_faceverify['img_y'][0])
print(adience_dataset_non_match_faceverify['img_x'][0])
print(adience_dataset_non_match_faceverify['match'])

adience_dataset_faceverify = pd.concat([adience_dataset_match_faceverify,adience_dataset_non_match_faceverify]).reset_index()
adience_dataset_faceverify = adience_dataset_faceverify.drop(labels=['index'], axis=1)
print(adience_dataset_faceverify)
adience_dataset_faceverify.to_csv("adience_dataset_facerecognition.csv", encoding="utf_8_sig")
'''

'''

celeba_ssd_path = '/Volumes/xiaozhe_SSD/datasets/celeba'

fn = partial(os.path.join, celeba_ssd_path) # csv检索用的

# 导入数据集的划分
list_eval_partition_data = pd.read_table(fn('list_eval_partition.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','partition'])

# 身份信息对应的图片信息
identity_CelebA = pd.read_table(fn('identity_CelebA.txt'), delim_whitespace=True, header=None, index_col=None, names = ['img','id'])
print(identity_CelebA)

mask = slice(141819, 202599, 1)
print(identity_CelebA[mask])

test_data_img_id = identity_CelebA[mask]

test_Data_imgx_id_imgy_match = pd.merge(test_data_img_id,test_data_img_id,on='id', how='left')
test_Data_imgx_id_imgy_match = test_Data_imgx_id_imgy_match.drop_duplicates(['img_x'], keep='last')
test_Data_imgx_id_imgy_match = test_Data_imgx_id_imgy_match.reset_index()
test_Data_imgx_id_imgy_match = test_Data_imgx_id_imgy_match.drop(labels='index', axis=1)
test_Data_imgx_id_imgy_match.insert(loc = 3, column='match', value=np.ones(test_Data_imgx_id_imgy_match.shape[0]))
test_Data_imgx_id_imgy_match = test_Data_imgx_id_imgy_match.drop(labels='id', axis=1)
print(test_Data_imgx_id_imgy_match)


img_random_sample = test_data_img_id['img'].sample(n=test_data_img_id.shape[0]).values
test_Data_imgx_id_imgy_non_match = test_data_img_id.reset_index()
test_Data_imgx_id_imgy_non_match = test_Data_imgx_id_imgy_non_match.drop(labels='index', axis=1)
test_Data_imgx_id_imgy_non_match.insert(loc = 2, column='img_y', value = img_random_sample)
test_Data_imgx_id_imgy_non_match = test_Data_imgx_id_imgy_non_match.drop(labels='id', axis=1)
test_Data_imgx_id_imgy_non_match.rename(columns={'img':'img_x'}, inplace=True)
test_Data_imgx_id_imgy_non_match.insert(loc = 2, column='match', value=np.zeros(test_Data_imgx_id_imgy_non_match.shape[0]))
print(test_Data_imgx_id_imgy_non_match)

celeba_facerecognition_test_dataset = pd.concat([test_Data_imgx_id_imgy_match, test_Data_imgx_id_imgy_non_match])
celeba_facerecognition_test_dataset = celeba_facerecognition_test_dataset.reset_index()
celeba_facerecognition_test_dataset = celeba_facerecognition_test_dataset.drop(labels='index', axis=1)
print(celeba_facerecognition_test_dataset)
# celeba_facerecognition_test_dataset.to_csv("celeba_face_verify_test_dataset.csv", encoding="utf_8_sig")
'''