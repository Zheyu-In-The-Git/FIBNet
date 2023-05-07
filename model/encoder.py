import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim, arcface_model):

        super(Encoder, self).__init__()

        # 将arcface的resnet部分中的layer[1],[2],[3]冻结住
        self.arcface_model_resnet50 = arcface_model.resnet50
        for name, param in self.arcface_model_resnet50.named_parameters():
            param.requires_grad_(False)

        for param in self.arcface_model_resnet50.layer4.parameters():
            param.requires_grad = True

        for param in self.arcface_model_resnet50.fc.parameters():
            param.requires_grad = True

        in_features = self.arcface_model_resnet50.fc.in_features

        self.arcface_model_resnet50.fc = nn.Linear(in_features, in_features)

        self.batchnorm = nn.BatchNorm1d(in_features)

        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.mu_fc = nn.Linear(in_features, latent_dim)

        self.log_var_fc = nn.Linear(in_features, latent_dim)

    def forward(self, x): # 输入的是表征
        x = self.arcface_model_resnet50(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


if __name__ == '__main__':
     #net = ResNet18Encoder(latent_dim=512, channels=3, act_fn='PReLU')
     #x = torch.randn(2, 3, 224, 224)
     #mu, log_var = net(x)
     #print( mu.shape, log_var.shape)


     #net = LitEncoder1(latent_dim=512, act_fn='PReLU')
     #x = torch.randn(2,3,224,224)
     #mu, log_var = net(x)
     #print(mu.shape, log_var.shape)

     from arcface_resnet50 import ArcfaceResnet50

     arcface_resnet50_net = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
     model = arcface_resnet50_net.load_from_checkpoint('/Users/xiaozhe/PycharmProjects/Bottleneck_Nets/lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/epoch=140-step=279350.ckpt')
     net = Encoder(latent_dim=512, arcface_model=arcface_resnet50_net)
     x = torch.randn(5, 3, 224, 224)
     mu, log_var = net(x)
     print(mu.shape, log_var.shape)














