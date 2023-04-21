import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super(LatentDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.BatchNorm1d(latent_dim//2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(latent_dim//2, latent_dim//2),
            nn.BatchNorm1d(latent_dim//2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(latent_dim//2, 1),

            #nn.Sigmoid()
        )
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

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    net = LatentDiscriminator(latent_dim=512)
    x = torch.randn(3, 512)
    # print(x.shape)

    # print(net)
    out = net(x)
    print(out.shape)