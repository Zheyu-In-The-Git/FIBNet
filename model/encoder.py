import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim):

        super(Encoder, self).__init__()

        self.encoder_net = nn.Sequential(
            nn.Linear(latent_dim , latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.mu_fc = nn.Linear(latent_dim * 2, latent_dim)
        self.log_var_fc = nn.Linear(latent_dim * 2, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # 输入的是表征
        x = self.encoder_net(x)
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

     net = Encoder(latent_dim=512)
     x = torch.randn(2,512)
     mu, log_var = net(x)
     print(mu.shape, log_var.shape)














