import matplotlib.backends.backend_pdf
import torch
import pytorch_lightning as pl
import os
import PIL
pl.seed_everything(83)

from matplotlib import pyplot as plt
from torchvision import transforms
# from arcface_resnet50 import ArcfaceResnet50
from sklearn.manifold import TSNE

from data import CelebaTSNEExperiment
from torch.utils.data import DataLoader
from model import BottleneckNets, Encoder, Decoder
import arcface_resnet50
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #******#

# Celeba数据集的准备
celeba_dir = 'D:\celeba'
celeba_dataset = CelebaTSNEExperiment(dim_img=224, data_dir=celeba_dir, sensitive_attr='Male', split='test_30%')
dataloader = DataLoader(celeba_dataset, batch_size=50, shuffle=True)



# 导入模型
#arcface_resnet50_net = arcface_resnet50.ArcfaceResnet50(in_features=1024, out_features=10177, s=64.0, m=0.50)
#model = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent1024\checkpoints\saved_model\face_recognition_resnet50\epoch=130-step=259400.ckpt').to(device)


# 导入Bottleneck模型

arcface_resnet50_net = arcface_resnet50.ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
#arcface = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
arcface = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\Administrator\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
encoder = Encoder(latent_dim=512, arcface_model=arcface)
decoder = Decoder(latent_dim=512, identity_nums=10177, s=64.0, m=0.50, easy_margin=False)
bottlenecknets = BottleneckNets(model_name='bottleneck', encoder=encoder, decoder=decoder, beta=0.1, batch_size=64, identity_nums=10177)
#model = bottlenecknets.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.0001\checkpoints\saved_models\last.ckpt', encoder=encoder,decoder=decoder)
model = bottlenecknets.load_from_checkpoint(r'C:\Users\Administrator\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.01\checkpoints\saved_models\last.ckpt',encoder=encoder,decoder=decoder)
model.to(device)

# 冻结住网络参数的梯度
for param in model.parameters():
    param.requires_grad_(False)

'''
   
# 让模型进行计算
Z_data = 0
S_data = 0
for i, item in enumerate(dataloader):
    print('i', i)
    x, u, s = item
    U, Z = model(x.to(device),u.to(device))
    #print(Z.shape)
    if i == 0:
        Z_data = Z
        S_data = s
    else:
        Z_data = torch.cat((Z_data, Z), 0)
        S_data = torch.cat((S_data, s), 0)
'''

Z_data = 0
S_data = 0
for i, item in enumerate(dataloader):
    print('i', i)
    x, u, s = item
    Z, _, _, _ = model(x.to(device),u.to(device))
    #print(Z.shape)
    if i == 0:
        Z_data = Z
        S_data = s
    else:
        Z_data = torch.cat((Z_data, Z), 0)
        S_data = torch.cat((S_data, s), 0)

Z_data = Z_data.cpu().detach().numpy()
S_data = S_data.cpu().detach().numpy()
print(Z_data.shape)
print(S_data.shape)
# 进行TSNE计算





tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=100.0, learning_rate=20.0, n_iter=4300,
            n_iter_without_progress=300, min_grad_norm=1e-07, metric='cosine', init='pca', verbose=1,
            random_state=83, method='barnes_hut', angle=0.5) # 这里metric可以换成cosine 到时候试一下  3*perplexity < nrow(data) - 1
X_tsne = tsne.fit_transform(Z_data)
print(X_tsne)


X_tsne_data = np.hstack((X_tsne, S_data))
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'gender'])
df_tsne.loc[df_tsne['gender'] == 0, 'gender'] = 'female'
df_tsne.loc[df_tsne['gender'] == 1, 'gender'] = 'male'

#markers = {"female": 'y', "male": "b"}
color = {"female": 'y', "male": "b"}

sns.scatterplot(data=df_tsne, hue='gender', x='Dim1', y='Dim2') # markers=markers
#sns.scatterplot(data=df_tsne, hue='gender') # 不打 x轴 和 y轴
plt.show()
# plt.savefig('arcface_512_resnet50.eps', format='eps', bbox_inches='tight') # 保存成.eps格式
