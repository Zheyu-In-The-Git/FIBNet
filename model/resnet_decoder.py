
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

identity_nums = 2622


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        print('one-hot',one_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels = identity_nums, act_fn = 'Softmax'):
        super(ResNetDecoder, self).__init__()
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

    #net = LitDecoder1(latent_dim=512, identity_nums=identity_nums, act_fn='Softmax')
    #x = torch.randn(2, 512)
    #out = net(x)
    #print(out.sum(dim = 1))

    arcface_metric = ArcMarginProduct(in_features=10, out_features=15)
    x = torch.randn(3, 10)
    label = torch.arange(3)
    out = arcface_metric(x, label)
    softmax = nn.Softmax(dim=1)
    soft_max_out = softmax(out)
    print('out', out)
    print('soft_max',soft_max_out)

