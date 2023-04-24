import torch
import pytorch_lightning as pl
import os
import PIL
pl.seed_everything(43)
from model import utility_discriminator
import torch.nn as nn
from model import ConstructBottleneckNets
from matplotlib import pyplot as plt
from torchvision import transforms
from arcface_resnet50 import ArcfaceResnet50
import torch.nn.functional as F



# 4月23日 celeba数据集训练arcface_resnet50
# 画
list = torch.load('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/data/arcface_confusion_cos.pt', map_location=torch.device('cpu'))
print(list.keys()) # fpr_cos, tpr_cos, thresholds_coss, eer_cos
print(list['fpr_cos'])
print('eer_cos:',list['eer_cos'])
fpr_cos = list['fpr_cos'].numpy()
tpr_cos = list['tpr_cos'].numpy()
thresholds_cos = list['thresholds_coss'].numpy()

plt.plot(fpr_cos, tpr_cos)
plt.xlabel("FPR",fontsize=15)
plt.ylabel("TPR",fontsize=15)

plt.title("ROC")
plt.legend(loc="lower right")
# plt.show()



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
img_x = PIL.Image.open(os.path.join("/Volumes/xiaozhe_SSD/datasets/celeba/img_align_celeba/img_align_celeba",'183019.jpg'))

img_y = PIL.Image.open(os.path.join("/Volumes/xiaozhe_SSD/datasets/celeba/img_align_celeba/img_align_celeba",'159664.jpg'))

x = trans(img_x)
y = trans(img_y)

batch = torch.stack([x, y], dim=0)
latent_x_y = model.resnet50(batch)

x_latent, y_latent = torch.chunk(latent_x_y, 2, dim=0)
x_latent = x_latent.squeeze(dim=0)
y_latent = y_latent.squeeze(dim=0)
cos_value = F.cosine_similarity(x_latent, y_latent)
print(cos_value)






