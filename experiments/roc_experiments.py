import torch
import pytorch_lightning as pl
import os
import PIL
pl.seed_everything(83)
from model import utility_discriminator
import torch.nn as nn
from matplotlib import pyplot as plt


# 预备测试的超参数beta 为 0.0001, 0.001, 0.01, 0.1, 1


# 5y月1日 celeba数据集训练arcface_resnet50
# -----------------ROC实验----------------------
#
list_arcface = torch.load(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.00001\bottleneck_roc_beta1e-05.pt', map_location=torch.device('cpu'))
list_bottleneck_beta_01 = torch.load(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.1\bottleneck_roc_beta0.1.pt', map_location=torch.device('cpu'))
list_bottleneck_beta_001 = torch.load(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.01\bottleneck_roc_beta0.01.pt', map_location=torch.device('cpu'))
list_bottleneck_beta_0001 = torch.load(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.001\bottleneck_roc_beta0.001.pt', map_location=torch.device('cpu'))
list_bottleneck_beta_00001 = torch.load(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.0001\bottleneck_roc_beta0.0001.pt', map_location=torch.device('cpu'))


# fpr相关数据
fpr_cos_arcface = list_arcface['fpr_cos'].numpy()
fpr_cos_bottleneck_beta_01 = list_bottleneck_beta_01['fpr_cos'].numpy()
fpr_cos_bottleneck_beta_001 = list_bottleneck_beta_001['fpr_cos'].numpy()
fpr_cos_bottleneck_beta_0001 = list_bottleneck_beta_0001['fpr_cos'].numpy()
fpr_cos_bottleneck_beta_00001 = list_bottleneck_beta_00001['fpr_cos'].numpy()

# tpr相关数据
tpr_cos_arcface = list_arcface['tpr_cos'].numpy()
tpr_cos_bottleneck_beta_01 = list_bottleneck_beta_01['tpr_cos'].numpy()
tpr_cos_bottleneck_beta_001 = list_bottleneck_beta_001['tpr_cos'].numpy()
tpr_cos_bottleneck_beta_0001 = list_bottleneck_beta_0001['tpr_cos'].numpy()
tpr_cos_bottleneck_beta_00001 = list_bottleneck_beta_00001['tpr_cos'].numpy()

# threshold相关数据
thresholds_cos_arcface = list_arcface['thresholds_cos'].numpy()
thresholds_cos_bottleneck_beta_01 = list_bottleneck_beta_01['thresholds_cos'].numpy()
thresholds_cos_bottleneck_beta_001 = list_bottleneck_beta_001['thresholds_cos'].numpy()
thresholds_cos_bottleneck_beta_0001 = list_bottleneck_beta_0001['thresholds_cos'].numpy()
thresholds_cos_bottleneck_beta_00001 = list_bottleneck_beta_00001['thresholds_cos'].numpy()


# 画线
plt.plot(fpr_cos_arcface, tpr_cos_arcface, linestyle='-', label='Arcface')
plt.plot(fpr_cos_bottleneck_beta_00001, tpr_cos_bottleneck_beta_00001, linestyle='-', label=r'$\beta$ = 0.0001')
plt.plot(fpr_cos_bottleneck_beta_0001, tpr_cos_bottleneck_beta_0001, linestyle='-', label=r'$\beta$ = 0.001')
plt.plot(fpr_cos_bottleneck_beta_001, tpr_cos_bottleneck_beta_001, linestyle='-', label=r'$\beta$ = 0.01')
plt.plot(fpr_cos_bottleneck_beta_01, tpr_cos_bottleneck_beta_01, linestyle='-', label=r'$\beta$ = 0.1')

plt.xlabel("FPR",fontsize=15)
plt.ylabel("TPR",fontsize=15)

plt.title("ROC")
plt.legend(loc="lower right")
plt.show()

print(list_arcface['eer_cos'])
print(list_bottleneck_beta_00001['eer_cos'])
print(list_bottleneck_beta_0001['eer_cos'])
print(list_bottleneck_beta_001['eer_cos'])
print(list_bottleneck_beta_01['eer_cos'])


'''
print(list.keys()) # fpr_cos, tpr_cos, thresholds_coss, eer_cos
print(list['fpr_cos'])
print('eer_cos:',list['eer_cos'])
fpr_cos = list['fpr_cos'].numpy()
tpr_cos = list['tpr_cos'].numpy()
thresholds_cos = list['thresholds_cos'].numpy()

# 之后可以用的


list2 = torch.load('data/arcface_confusion_cos.pt', map_location=torch.device('cpu'))
# print(list2.keys()) # (['fpr_cos', 'tpr_cos', 'thresholds_coss', 'eer_cos'])
fpr_cos_wrong = list2['fpr_cos'].numpy()
tpr_cos_wrong = list2['tpr_cos'].numpy()
thresholds_cos_wrong = list2['thresholds_coss'].numpy()
'''






'''

# ------ 测人脸# ----
trans = transforms.Compose([transforms.CenterCrop((130, 130)),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                                    ])

# load_path = torch.load('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_models/face_recognition_resnet50', map_location=torch.device('cpu'))
# print(load_path['state_dict'])
#model = load_path['state_dict']

arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
# model = arcface_resnet50_net.load_from_checkpoint('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_models/face_recognition_resnet50')
model = arcface_resnet50_net.load_from_checkpoint('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/last.ckpt')
img_x = PIL.Image.open(os.path.join("/Users/xiaozhe/datasets/celeba/img_align_celeba",'145306.jpg'))

img_y = PIL.Image.open(os.path.join("/Users/xiaozhe/datasets/celeba/img_align_celeba",'171994.jpg'))

x = trans(img_x)
y = trans(img_y)

batch = torch.stack([x, y], dim=0)
latent_x_y = model.resnet50(batch)

x_latent, y_latent = torch.chunk(latent_x_y, 2, dim=0)
x_latent = x_latent.reshape(x_latent.shape[0], -1)
y_latent = y_latent.reshape(x_latent.shape[0], -1)
cos_value = F.cosine_similarity(x_latent, y_latent)
print(cos_value)

'''