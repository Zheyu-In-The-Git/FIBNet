
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

identity_nums = 2622

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels = identity_nums, act_fn = 'Softmax'):
        super(ResNetDecoder, self).__init__()

        if act_fn == 'ReLU':
            self.act_fn = nn.ReLU()

        if act_fn == 'PReLU':
            self.act_fn = nn.PReLU()

        if act_fn == 'Softmax':
            self.act_fn = nn.Softmax()

        self.net = nn.Sequential(

            nn.Linear(latent_dim,  latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(latent_dim, num_channels),
            #self.act_fn # TODO：使用Crossentropy损失的时候似乎不需要使用
        )


    def forward(self, x):
        x = self.net(x)
        return x



class LitDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels = identity_nums, act_fn = 'Softmax'):
        super(LitDecoder, self).__init__()

        if act_fn == 'Softmax':
            self.act_fn = nn.Softmax(dim=1)

        self.net = nn.Sequential(

            nn.Linear(latent_dim, 4 * latent_dim),
            nn.BatchNorm1d(4 * latent_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(4 * latent_dim, 4 * latent_dim),
            nn.BatchNorm1d(4 * latent_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(4 * latent_dim, num_channels),
            self.act_fn
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

    def forward(self,x):
        x = self.net(x)
        return x


def ResNet50Decoder(latent_dim, identity_nums, act_fn = 'Softmax'):
    return ResNetDecoder(latent_dim=latent_dim, num_channels=identity_nums, act_fn = act_fn)

def ResNet101Decoder(latent_dim, identity_nums, act_fn = 'Softmax'):
    return ResNetDecoder(latent_dim=latent_dim, num_channels=identity_nums, act_fn = act_fn)

def ResNet18Decoder(latent_dim, identity_nums, act_fn = 'Softmax'):
    return ResNetDecoder(latent_dim=latent_dim, num_channels=identity_nums, act_fn = act_fn)

def LitDecoder1(latent_dim, identity_nums, act_fn = 'Softmax'):
    return LitDecoder(latent_dim=latent_dim, num_channels=identity_nums, act_fn=act_fn)


if __name__ == '__main__':
    # net = ResNet50Decoder(latent_dim=1024, identity_nums=identity_nums, act_fn='Softmax')
    # x = torch.randn(2, 1024)
    # out = net(x)
    # print(out.shape)

    net = LitDecoder1(latent_dim=512, identity_nums=identity_nums, act_fn='Softmax')
    x = torch.randn(2, 512)
    out = net(x)
    print(out.sum(dim = 1))
