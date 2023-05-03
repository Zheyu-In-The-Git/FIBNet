
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

identity_nums = 2622


class Decoder(nn.Module):
    def __init__(self, latent_dim, identity_nums, s, m, easy_margin=False): # s = 64.0, m =0.5
        super(Decoder, self).__init__()
        self.in_features = latent_dim
        self.out_features = identity_nums
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(identity_nums, latent_dim))
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
        # print('one-hot', one_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------

        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output



if __name__ == '__main__':
    # net = ResNet50Decoder(latent_dim=1024, identity_nums=identity_nums, act_fn='Softmax')
    # x = torch.randn(2, 1024)
    # out = net(x)
    # print(out.shape)


    decoder = Decoder(latent_dim=512, identity_nums=10177, s = 64.0, m=0.5, easy_margin=False)
    x = torch.rand(3, 512)
    label = torch.arange(3)
    out = decoder(x, label)
    softmax = nn.Softmax(dim=1)
    soft_max_out = softmax(out)
    print('out', out)
    print('soft_max', soft_max_out.argmax(axis=1))

