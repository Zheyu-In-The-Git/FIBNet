
from deepface import DeepFace
import numpy as np
import pandas as pd
from functools import partial
import os
from torchvision import transforms
import PIL


def traverse_numbers():
    numbers = []
    for i in range(500):
        numbers.append("{:03d}".format(i))
    numbers = np.array(numbers)
    return numbers

def list_files(directory):
    """
    遍历指定文件夹中的所有文件并打印它们的路径
    """
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            path_list.append(file_path)
            #print(file_path)
    return path_list





identity_list = traverse_numbers()

'''


CasiaFace_path = 'E:\datasets\CASIA-FaceV5'

fn = partial(os.path.join, CasiaFace_path)

gender_list = []
for i in identity_list:
    identity_path = list_files(fn(i))[0]
    #print(identity_path)
    try:
        # 尝试分析图像中的性别
        result = DeepFace.analyze(img_path=identity_path, actions=['gender'])
        if result[0]['dominant_gender'] == 'Man':
            gender_list.append(1)
        else:
            gender_list.append(0)

    except ValueError as e:
        # 处理面部未检测到的异常
        gender_list.append(0)
        print('the except appears' + identity_path)




np.savez('gender_data', my_list = gender_list)
'''

'''

gender_data = np.load('E:\Bottleneck_Nets\data\csv_file\gender_data.npz')
gender_list = gender_data['my_list']
gender_list[113] = 0
gender_list[137] = 1
gender_list[254] = 0
gender_list[257] = 0
gender_list[277] = 1
gender_list[309] = 1
gender_list[339] = 0
gender_list[352] = 1
gender_list[419] = 0
gender_list[460] = 0
gender_list[464] = 0
print(gender_list)
#np.savez('gender_data_real', my_list = gender_list)
'''

'''

gender_data = np.load('E:\Bottleneck_Nets\data\csv_file\gender_data_real.npz')
gender_list = gender_data['my_list']
print(gender_list)


traverse_list = np.array(traverse_numbers())
genderlist = np.array(gender_list)
concat_id_gender = np.column_stack((traverse_list, genderlist))
id_gender_pd = pd.DataFrame(concat_id_gender, columns=['identity', 'gender'])


image_idx_list = np.load(r'E:\Bottleneck_Nets\data\csv_file\image_idx_list.npy')
id_subpath_pd = pd.DataFrame(image_idx_list, columns = ['identity','sub_img_path'])
print(id_subpath_pd)

merged_df = pd.merge(id_subpath_pd, id_gender_pd, on='identity', how='inner')
print(merged_df)
merged_df.to_csv("CasiaFace_id_subpath_gender.csv", encoding="utf_8_sig")

#array = np.array([traverse_numbers(), np.ones(500)])
#print(array.transpose().shape)
#np.savez('mydata', my_list = array.transpose())

#result = DeepFace.analyze(img_path=r'E:\datasets\CASIA-FaceV5\000\000_0.bmp', actions=['gender'])
#print(result[0]['dominant_gender'])

#identity_gender = pd.DataFrame([traverse_numbers(), np.ones(500)], columns=['identity','gender'])
#print(identity_gender)

'''


csv_file_path = 'E:\datasets\CASIA-FaceV5\csv_file'
casia_jpg_path = 'E:\datasets\CASIA-FaceV5\dataset_jpg\dataset'

casia_fn = partial(os.path.join, casia_jpg_path)
csv_file_path_fn = partial(os.path.join, csv_file_path)

trans = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])

img_idx_csv_path = csv_file_path_fn('image_idx_list.npy')
print(img_idx_csv_path)
img_idx_list = np.load(img_idx_csv_path)
print(img_idx_list)


img_path = casia_fn(img_idx_list[10][0], img_idx_list[10][1])
img_path = img_path.replace('.bmp', '.jpg')
print(img_path)

x = PIL.Image.open(img_path)
x = trans(x)

x_processed_pil = transforms.ToPILImage()(x)

x_processed_pil.show()


