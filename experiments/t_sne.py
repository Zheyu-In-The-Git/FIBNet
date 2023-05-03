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
import arcface_resnet50
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Celeba数据集的准备
celeba_dir = '/Users/xiaozhe/datasets/celeba'
celeba_dataset = CelebaTSNEExperiment(dim_img=224, data_dir=celeba_dir, sensitive_attr='Male', split='test_30%')
dataloader = DataLoader(celeba_dataset, batch_size=550, shuffle=True)



# 导入模型
arcface_resnet50_net = arcface_resnet50.ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
model = arcface_resnet50_net.load_from_checkpoint('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=140-step=279350.ckpt')

# 让模型进行计算
Z_data = 0
S_data = 0
for i, item in enumerate(dataloader):
    print('i', i)
    x, u, s = item
    U, Z = model(x,u)
    #print(Z.shape)
    Z_data = Z
    S_data = s
    break
Z_data = Z_data.detach().numpy()
S_data = S_data.detach().numpy()
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

sns.scatterplot(data=df_tsne, hue='gender', x='Dim1', y='Dim2')
plt.show()
