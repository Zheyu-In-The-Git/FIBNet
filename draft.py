
import torch
import pytorch_lightning as pl
pl.seed_everything(43)
from model import utility_discriminator
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms
'''
# 二元交叉熵
x = torch.Tensor([0.1, 0.5, 0.4])
y = torch.Tensor([0,1,0])

bce = nn.BCELoss()
dis_bce = bce(x,y)
print(dis_bce)
total = -torch.log(torch.Tensor([0.5])) -torch.log(torch.Tensor([0.9])) -torch.log(torch.Tensor([0.6]))
print(total/3)


x = torch.Tensor([1, 2, 3])
sigmoid = nn.Sigmoid()
print(sigmoid(x))
'''
'''
# 写一下detach()

a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
a = a.detach() # 会将requires_grad 属性设置为False
print(a.requires_grad)
'''
'''

# 考察一下torchmetrics
import torchmetrics
preds = torch.randn(10, 5).softmax(dim=-1)
print(preds.shape)
target = torch.randint(5, (10,))
print(target.shape)

acc = torchmetrics.functional.accuracy(preds, target)
print(acc)
'''
# import torch
# print(torch.__version__)



# a = torch.tensor([1., 2., 3.], requires_grad = True)
# b = a.clone()
#
# print(a.data_ptr())
# print(b.data_ptr())
#
# print(a)
# print(b)
# print('-'*30)
#
# c = a * 2
# d = b * 3
#
# c.sum().backward()
# print(a.grad)
#
# d.sum().backward()
# print(a.grad)
# print(b.grad)
#
# print('-'*60)
'''


a = torch.tensor([1., 2., 3.],requires_grad=True)
b = a.detach()

print(a.data_ptr()) # 2432102290752
print(b.data_ptr()) # 2432102290752 # 内存位置相同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.]) # 这里为False，就省略了
print('-!'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

# d.sum().backward()
print(a.grad) # 报错了！ 由于b不记录计算图，因此无法计算b的相关梯度信息
print(b.grad)

print( 'gpu count: ',torch.cuda.device_count())


print(pl.__version__)
'''



'''
# 审查模型参数
pretrained_filename = 'lightning_logs/bottleneck_test_version_1/checkpoints/saved_models/epoch=0-step=400.ckpt'
bottlenecknets = ConstructBottleneckNets(args)
model = bottlenecknets.load_from_checkpoint(pretrained_filename)

x = torch.randn(2, 3, 224, 224)
z, u_hat, s_hat, u_value, s_value, mu, log_var = model(x)
print(z, mu, log_var)
'''
'''
sigmoid = nn.Sigmoid()
def kl_estimate_value(discriminating):
    discriminated = sigmoid(discriminating)
    kl_estimate_value = (torch.log(discriminated) - torch.log(1 - discriminated)).sum(1).mean()
    return kl_estimate_value.detach()


net = utility_discriminator.UtilityDiscriminator(utility_dim=10177)
x = torch.randn(3, 10177)
out = net(x)
print(out)

value = kl_estimate_value(out)
print(value)
'''
'''

from torchmetrics.classification import ConfusionMatrix # 混淆矩阵

import platform


def get_platform():
    sys_platform = platform.platform().lower()
    print(sys_platform)
    if "darwin-22.1.0-x86_64-i386-64bit" in sys_platform:
        print('Mac os')
    else:
        print('Windows')
get_platform()

'''
'''


# torch.save()
list = torch.load('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/data/arcface_confusion_cos.pt', map_location=torch.device('cpu'))
print(list.keys()) # fpr_cos, tpr_cos, thresholds_coss, eer_cos
print(list['fpr_cos'])
fpr_cos = list['fpr_cos'].numpy()
tpr_cos = list['tpr_cos'].numpy()
thresholds_cos = list['thresholds_coss'].numpy()

plt.plot(fpr_cos, tpr_cos)
plt.xlabel("FPR",fontsize=15)
plt.ylabel("TPR",fontsize=15)

plt.title("ROC")
plt.legend(loc="lower right")
# plt.show()


trans = transforms.Compose([transforms.CenterCrop((130, 130)),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                                    ])

from collections import OrderedDict
new_state_dict = OrderedDict()

load_path = torch.load('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_models/face_recognition_resnet50', map_location=torch.device('cpu'))
print(load_path['state_dict'])
model = load_path['state_dict']

#x = torch.randn( 3, 224, 224)
#output = model(x)
#print(output)

'''
'''

import numpy as np
s = np.sqrt(2) * np.log(10177-1)
print(s)
'''

'''

from arcface_resnet50 import ArcfaceResnet50
arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
model = arcface_resnet50_net.load_from_checkpoint('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=140-step=279350.ckpt')
#print(model)

#model_resnet50 = model.resnet50.layer4[2] # 这是最后一层卷积
#print(model_resnet50)


#for name, param in model.named_parameters():
#    param.requires_grad = False
    #if model.resnet50.layer4[2] in name:
    #    param.requires_grad =True


for param in model.resnet50.parameters():
    param.requires_grad = False

for param in model.resnet50.layer4.parameters():
    param.requires_grad = True

for param in model.resnet50.fc.parameters():
    param.requires_grad = True

print(model.resnet50.fc.in_features)

for name, param in model.named_parameters():
    if not param.requires_grad:
        print(name, 'is frozen.')
    else:
        print(name, 'is trainable.')
        
'''

'''

from facenet_pytorch import MTCNN
import cv2
import PIL
from torchvision import transforms
import torchvision.transforms.functional as F


mtcnn = MTCNN(keep_all=True)
#img = cv2.imread('/Users/xiaozhe/datasets/celeba/img_align_celeba/img_align_celeba/000001.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

trans = transforms.Compose([transforms.CenterCrop((180, 180)),
                            transforms.RandomHorizontalFlip(p=0.5)])

img = PIL.Image.open('/Users/xiaozhe/datasets/celeba/img_align_celeba/img_align_celeba/001005.jpg')
img = trans(img)

boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

# print(f"Detected {len(boxes)} faces")

max_prob_idx = probs.argmax()
max_prob_box = boxes[max_prob_idx]

x1, y1, x2, y2 = max_prob_box.astype(int)

cropped_face_position = (x1, y1, x2, y2)
h = y2 - y1
w = x2 - x1

cropped_face = F.crop(img, x1, y1, h, w)
trans2 = transforms.Compose([
    transforms.Resize((112, 112)),
])
cropped_face = trans2(cropped_face)

#print(cropped_face.shape)
cropped_face.show()
# img.show()
'''

'''
# 在图像上绘制人脸框和关键点
for box in boxes:
    x1, y1, x2, y2 = box.astype(int)
    cropped_face_position = (x1, y1, x2, y2)
    h = y2 - y1
    w = x2 - x1
    cropped_face = F.crop(img, x1, y1, h, w)
    cropped_face.show()
'''
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('lightning_logs/model_ir_se50.pth', map_location=device)
print(model)
'''
'''

import pickle
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn

#model = models.resnet50()

#weights = torch.load('lightning_logs/resnet50_scratch_weight.pkl')
#model.load_state_dict(weights)
#print(model)
#torch.save(model.state_dict(), 'resnet50.pth')

#with open('lightning_logs/resnet50_scratch_weight.pkl', 'rb') as f:
#    model = pickle.load(f)

# torch.save(model.state_dict(), 'resnet-50.pth')

with open('lightning_logs/resnet50_scratch_weight.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# 转换为PyTorch模型
state_dict = {}
for k, v in model_dict.items():
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    state_dict[k] = v
model = models.resnet50()
model.fc = nn.Linear(2048, 8631)
# print(model.fc.out_features)
model.load_state_dict(state_dict)
print(model)

model.fc = nn.Linear(2048, 512)
print(model)
'''



'''

import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim, arcface_model):

        super(Encoder, self).__init__()

        # 将arcface的resnet部分中的layer[1],[2],[3]冻结住
        self.arcface_model_resnet50 = arcface_model.resnet50
        #for name, param in self.arcface_model_resnet50.named_parameters():
        #    param.requires_grad_(False)

        #for param in self.arcface_model_resnet50.layer4.parameters():
        #    param.requires_grad = True

        #for param in self.arcface_model_resnet50.fc.parameters():
        #    param.requires_grad = True

        in_features = self.arcface_model_resnet50.fc.in_features

        self.arcface_model_resnet50.fc = nn.Linear(in_features, 512)

        self.batchnorm512 = nn.BatchNorm1d(512)

        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.mu_fc = nn.Linear(512, latent_dim)

        self.log_var_fc = nn.Linear(512, latent_dim)

    def forward(self, x): # 输入的是表征
        x = self.arcface_model_resnet50(x)
        x = self.batchnorm512(x)
        x = self.leakyrelu(x)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


if __name__ == '__main__':

     from arcface_resnet50 import *
     arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
     #model = arcface_resnet50_net.load_from_checkpoint('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=140-step=279350.ckpt')
     net = Encoder(latent_dim=512, arcface_model=arcface_resnet50_net)
     print(net)
     x = torch.randn(5, 3, 224, 224)
     mu, log_var = net(x)
     print(mu, log_var)
'''

import numpy as np
print(pl.__version__)
print(np.__version__)
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print('hello')
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
print('hello world!!')


import os
print(os.getcwd()) # 工作路径
print(os.path.abspath('.'))

path = "/data/qbx_20210083/datasets" #文件夹目录
files= os.listdir(path)
print(files)

