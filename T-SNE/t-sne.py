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


from tsne_data import CelebaTSNEExperiment, CelebaTSNERaceExperiment, LFWTSNEExperiment, AdienceTSNEGenderExperiment, AdienceTSNERaceExperiment
from torch.utils.data import DataLoader
from model import BottleneckNets, Encoder, Decoder
import arcface_resnet50
import numpy as np
import pandas as pd
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")





def TSNEExperiment(model_name, sensitive_attr, dataloader, data_name):
    ###############################################
    ########### Arcface 实验部分呢 ###################
    ###############################################

    if model_name == 'Arcface':
        arcface_resnet50_net = arcface_resnet50.ArcfaceResnet50(in_features=1024, out_features=10177, s=64.0, m=0.50)
        model = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\epoch=48-step=95800.ckpt').to(device)

        for param in model.parameters():
            param.requires_grad_(False)


        # 数据加载


        Z_data = 0
        S_data = 0
        for i, item in enumerate(dataloader):
            print('i', i)
            x, u, s = item
            U, Z = model(x.to(device), u.to(device))
            # print(Z.shape)
            if i == 0:
                Z_data = Z
                S_data = s
            else:
                Z_data = torch.cat((Z_data, Z), 0)
                S_data = torch.cat((S_data, s), 0)

        # 计算tsne
        tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=100.0, learning_rate=20.0, n_iter=4300,
                    n_iter_without_progress=300, min_grad_norm=1e-07, metric='cosine', init='pca', verbose=1,
                    random_state=89, method='barnes_hut',
                    angle=0.5)  # 这里metric可以换成cosine 到时候试一下  3*perplexity < nrow(data) - 1
        X_tsne = tsne.fit_transform(Z_data.cpu())

        print(X_tsne)




        # 保存tsne数据，以及对应的s
        X_tsne_data = np.hstack((X_tsne, S_data))
        np.savez('t-sne'+'_'+ sensitive_attr +'_' + model_name +'_'+ data_name +'.npz', X_tsne_data=X_tsne_data)  # 可能就是前500个是1，后500个是0
        # 打开查看用np.load(t-sne.npz)


        # 查看图片和保存pdf文件
        if sensitive_attr == 'Male':

            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'sensitive_attr'])
            df_tsne.loc[df_tsne['sensitive_attr'] == 0, 'sensitive_attr'] = 'female'
            df_tsne.loc[df_tsne['sensitive_attr'] == 1, 'sensitive_attr'] = 'male'

            sns.scatterplot(data=df_tsne, hue='sensitive_attr', x='Dim1', y='Dim2')  # markers=markers
            # sns.scatterplot(data=df_tsne, hue='gender') # 不打 x轴 和 y轴

            plt.title('t-sne'+'_'+ sensitive_attr +'_' + model_name +'_'+ data_name +'.npz')
            plt.savefig('t-sne'+'_'+ sensitive_attr +'_' + model_name +'_'+ data_name +'.pdf')
            plt.show()
            plt.close()

        elif sensitive_attr == 'White':

            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'sensitive_attr'])
            df_tsne.loc[df_tsne['sensitive_attr'] == 0, 'sensitive_attr'] = 'colored_people'
            df_tsne.loc[df_tsne['sensitive_attr'] == 1, 'sensitive_attr'] = 'white'

            sns.scatterplot(data=df_tsne, hue='sensitive_attr', x='Dim1', y='Dim2')  # markers=markers
            # sns.scatterplot(data=df_tsne, hue='gender') # 不打 x轴 和 y轴

            plt.title('t-sne' + '_' + sensitive_attr + '_' + model_name + '_' + data_name + '.npz')
            plt.savefig('t-sne'+'_'+ sensitive_attr +'_' + model_name +'_'+ data_name +'.pdf')
            plt.show()
            plt.close()




    #####################################################
    ########### Bottleneck 0.1 实验部分 ##################
    #####################################################
    elif model_name == 'Bottleneck_0.1':
        arcface_resnet50_net = arcface_resnet50.ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
        arcface = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
        #arcface = arcface_resnet50_net.load_from_checkpoint(r'C:\Users\Administrator\PycharmProjects\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
        encoder = Encoder(latent_dim=512, arcface_model=arcface)
        decoder = Decoder(latent_dim=512, identity_nums=10177, s=64.0, m=0.50, easy_margin=False)
        bottlenecknets = BottleneckNets(model_name='bottleneck', encoder=encoder, decoder=decoder, beta=0.1,batch_size=64, identity_nums=10177)
        model = bottlenecknets.load_from_checkpoint(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.1\checkpoints\saved_models\last.ckpt', encoder=encoder,decoder=decoder)
        #model = bottlenecknets.load_from_checkpoint(r'C:\Users\Administrator\PycharmProjects\Bottleneck_Nets\lightning_logs\bottleneck_experiment_latent_new_512_beta0.1\checkpoints\saved_models\last.ckpt', encoder=encoder, decoder=decoder)
        model.to(device)

        for param in model.parameters():
            param.requires_grad_(False)



        Z_data = 0
        S_data = 0
        for i, item in enumerate(dataloader):
            print('i', i)
            x, u, s = item
            Z, _, _, _ = model(x.to(device), u.to(device))
            # print(Z.shape)
            if i == 0:
                Z_data = Z
                S_data = s
            else:
                Z_data = torch.cat((Z_data, Z), 0)
                S_data = torch.cat((S_data, s), 0)


        tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=100.0, learning_rate=20.0, n_iter=4300,
                    n_iter_without_progress=300, min_grad_norm=1e-07, metric='cosine', init='pca', verbose=1,
                    random_state=89, method='barnes_hut',
                    angle=0.5)  # 这里metric可以换成cosine 到时候试一下  3*perplexity < nrow(data) - 1
        X_tsne = tsne.fit_transform(Z_data.cpu())

        print(X_tsne)

        X_tsne_data = np.hstack((X_tsne, S_data))
        np.savez('t-sne'+'_'+ sensitive_attr +'_' + model_name +'_'+ data_name +'.npz', X_tsne_data=X_tsne_data)  # 可能就是前500个是1，后500个是0
        # 打开查看用np.load(t-sne.npz)





        # 查看图片和保存pdf文件
        if sensitive_attr == 'Male':

            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'sensitive_attr'])
            df_tsne.loc[df_tsne['sensitive_attr'] == 0, 'sensitive_attr'] = 'female'
            df_tsne.loc[df_tsne['sensitive_attr'] == 1, 'sensitive_attr'] = 'male'

            sns.scatterplot(data=df_tsne, hue='sensitive_attr', x='Dim1', y='Dim2')  # markers=markers
            # sns.scatterplot(data=df_tsne, hue='gender') # 不打 x轴 和 y轴

            plt.title('t-sne' + '_' + sensitive_attr + '_' + model_name + '_' + data_name + '.npz')
            plt.savefig('t-sne' + '_' + sensitive_attr + '_' + model_name + '_' + data_name + '.pdf')
            plt.show()
            plt.close()

        elif sensitive_attr == 'White':

            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'sensitive_attr'])
            df_tsne.loc[df_tsne['sensitive_attr'] == 0, 'sensitive_attr'] = 'colored_people'
            df_tsne.loc[df_tsne['sensitive_attr'] == 1, 'sensitive_attr'] = 'white'

            sns.scatterplot(data=df_tsne, hue='sensitive_attr', x='Dim1', y='Dim2')  # markers=markers
            # sns.scatterplot(data=df_tsne, hue='gender') # 不打 x轴 和 y轴

            plt.title('t-sne' + '_' + sensitive_attr + '_' + model_name + '_' + data_name + '.npz')
            plt.savefig('t-sne' + '_' + sensitive_attr + '_' + model_name + '_' + data_name + '.pdf')
            plt.show()
            plt.close()









if __name__ == '__main__':
    ##########################################################
    #################   数据集准备部分  #########################
    ##########################################################

    # 数据集部分
    celeba_dir = 'D:\datasets\celeba'
    lfw_dir = 'D:\datasets\lfw\lfw112'
    adience_dir = 'D:\datasets\Adience'

    # 性别
    celeba_gender_dataset = CelebaTSNEExperiment(dim_img=224, data_dir=celeba_dir, sensitive_attr='Male', split='test_30%')
    celeba_gender_dataloader = DataLoader(celeba_gender_dataset, batch_size=50, shuffle=True)

    lfw_gender_dataset = LFWTSNEExperiment(dim_img=224, data_dir=lfw_dir, sensitive_attr='Male', split='all', img_path_replace=True)
    lfw_gender_dataloader = DataLoader(lfw_gender_dataset, batch_size=50, shuffle=True)

    adience_gender_dataset = AdienceTSNEGenderExperiment(dim_img=224, data_dir=adience_dir,  sensitive_attr='Male')
    adience_gender_dataloader = DataLoader(adience_gender_dataset, batch_size=50, shuffle=True)


    # 种族
    celeba_race_dataset = CelebaTSNERaceExperiment(dim_img=224, data_dir=celeba_dir, split='test_30%')
    celeba_race_dataloader = DataLoader(celeba_race_dataset, batch_size=50, shuffle=True)

    lfw_race_dataset =LFWTSNEExperiment(dim_img=224, data_dir=lfw_dir, sensitive_attr='White', split='all', img_path_replace=True)
    lfw_race_dataloader = DataLoader(lfw_race_dataset, batch_size=50, shuffle=True)

    adience_race_dataset = AdienceTSNERaceExperiment(dim_img=224, data_dir=adience_dir, identity_nums=10177)
    adience_race_dataloader = DataLoader(adience_race_dataset, batch_size=50, shuffle=True)




    #############################################################
    #################   Arcface 实验部分  #########################
    #############################################################

    # 性别
    TSNEExperiment(model_name='Arcface', sensitive_attr='Male', dataloader=celeba_gender_dataloader, data_name='celeba')
    TSNEExperiment(model_name='Arcface', sensitive_attr='Male', dataloader=lfw_gender_dataloader, data_name='lfw')
    TSNEExperiment(model_name='Arcface', sensitive_attr='Male', dataloader=adience_gender_dataloader, data_name='adience')

    # 种族
    TSNEExperiment(model_name='Arcface', sensitive_attr='White', dataloader=celeba_race_dataloader, data_name='celeba')
    TSNEExperiment(model_name='Arcface', sensitive_attr='White', dataloader=lfw_race_dataloader, data_name='lfw')
    TSNEExperiment(model_name='Arcface', sensitive_attr='White', dataloader=adience_race_dataloader, data_name='adience')




    #############################################################
    #################   bottleneck 实验部分  #########################
    #############################################################

    # 性别
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='Male', dataloader=celeba_gender_dataloader, data_name='celeba')
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='Male', dataloader=lfw_gender_dataloader, data_name='lfw')
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='Male', dataloader=adience_gender_dataloader,data_name='adience')

    # 种族
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='White', dataloader=celeba_race_dataloader, data_name='celeba')
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='White', dataloader=lfw_race_dataloader, data_name='lfw')
    TSNEExperiment(model_name='Bottleneck_0.1', sensitive_attr='White', dataloader=adience_race_dataloader, data_name='adience')























