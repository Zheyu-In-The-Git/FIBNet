import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim, arcface_model):

        super(Encoder, self).__init__()

        self.arcface_model_resnet50 = arcface_model.resnet50

        in_features = self.arcface_model_resnet50.fc.in_features

        self.arcface_model_resnet50.fc = nn.Linear(in_features, 512)

        self.batchnorm512 = nn.BatchNorm1d(512)

        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.mu_fc = nn.Linear(512, latent_dim)

        self.log_var_fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.arcface_model_resnet50(x)
        x = self.batchnorm512(x)
        x = self.leakyrelu(x)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


if __name__ == '__main__':

     from arcface_resnet50 import *
     arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)

     net = Encoder(latent_dim=512, arcface_model=arcface_resnet50_net)
     print(net)
     x = torch.randn(5, 3, 224, 224)
     mu, log_var = net(x)
     print(mu, log_var)














