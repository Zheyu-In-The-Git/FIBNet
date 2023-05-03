import torch
import pytorch_lightning as pl
import os
import PIL
pl.seed_everything(83)
from model import utility_discriminator
import torch.nn as nn
from model import ConstructBottleneckNets
from matplotlib import pyplot as plt
from torchvision import transforms
from arcface_resnet50 import ArcfaceResnet50
from sklearn.manifold import TSNE
import torch.nn.functional as F

# 预备测试的超参数beta 为 0.0001， 0.1， 0.3， 0.5， 0.8， 0.95


# 5y月1日 celeba数据集训练arcface_resnet50
# -----------------ROC实验----------------------
#
list = torch.load('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/lightning_log/roc_arcface_celeba_512.pt', map_location=torch.device('cpu'))
print(list.keys()) # fpr_cos, tpr_cos, thresholds_coss, eer_cos
print(list['fpr_cos'])
print('eer_cos:',list['eer_cos'])
fpr_cos = list['fpr_cos'].numpy()
tpr_cos = list['tpr_cos'].numpy()
thresholds_cos = list['thresholds_cos'].numpy()

# 之后可以用的
'''

list2 = torch.load('data/arcface_confusion_cos.pt', map_location=torch.device('cpu'))
# print(list2.keys()) # (['fpr_cos', 'tpr_cos', 'thresholds_coss', 'eer_cos'])
fpr_cos_wrong = list2['fpr_cos'].numpy()
tpr_cos_wrong = list2['tpr_cos'].numpy()
thresholds_cos_wrong = list2['thresholds_coss'].numpy()
'''


plt.plot(fpr_cos, tpr_cos, linestyle='-')
# plt.plot(fpr_cos_wrong, tpr_cos_wrong, linestyle='-.')

plt.xlabel("FPR",fontsize=15)
plt.ylabel("TPR",fontsize=15)

plt.title("ROC")
plt.legend(loc="lower right")
plt.show()





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