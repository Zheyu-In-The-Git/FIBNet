import torch
import torch.nn as nn
import torch.nn.functional as F
# from ResNet import Bottleneck
import math

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample = None, stride = 1, act_fn = 'ReLU'):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding =0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride

        if act_fn == 'ReLU':
            self.act_fn = nn.ReLU()

        if act_fn == 'PReLU':
            self.act_fn = nn.PReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.act_fn(self.batch_norm1(self.conv1(x)))

        x = self.act_fn(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)

        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x += identity
        x = self.act_fn(x)

        return x



class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=False, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x




class ResNetEncoder(nn.Module):
    def __init__(self, ResBlock, layer_list, latent_dim, num_channels, act_fn = 'ReLU'):
        '''
        ResNet的基本结构
        :param ResBlock:
        :param layer_list: 可以参照何恺明的论文 表1
        :param num_classes: 最终分类的数量，Celeb的个体数量
        :param num_channels: 3 RGB
        :param act_fn: 激活函数，后面可能换成PReLU函数
        '''

        super(ResNetEncoder, self).__init__()
        # 这些都是开辟的网络结构，真正的运算过程在forward函数里面
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm1d = nn.BatchNorm1d(latent_dim)

        if act_fn == 'ReLU':
            self.act_fn = nn.ReLU()

        if act_fn == 'PReLU':
            self.act_fn = nn.PReLU()

        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes = 64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes = 128, stride = 2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes = 256, stride = 2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes = 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.mu_fc1 = nn.Linear(512 * ResBlock.expansion, latent_dim)
        self.mu_fc2 = nn.Linear(latent_dim, latent_dim)

        self.log_var_fc1 = nn.Linear(512 * ResBlock.expansion, latent_dim)
        self.log_var_fc2 = nn.Linear(latent_dim, latent_dim)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.2, True)

        # 模型初始化
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
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(x)

        # mu的网络结构设计
        mu = self.mu_fc1(x)
        mu = self.batch_norm1d(mu)
        mu = self.leakyrelu(mu)
        mu = self.mu_fc2(mu)

        # log_var的网络结构设计
        log_var = self.log_var_fc1(x)
        log_var = self.batch_norm1d(log_var)
        log_var = self.leakyrelu(log_var)
        log_var = self.log_var_fc2(log_var)
        return mu, log_var

        # TODO: 这里只输出均值和对数方差，还没做重参数化技巧, 重参数化技巧准备设计总体网络的时候写
        # TODO: 重参数化技巧的网络结构可以参考 https://zhuanlan.zhihu.com/p/452743042


    # 4个层用的子网络模块
    def _make_layer(self, ResBlock, blocks, planes, stride = 1):
        ii_downsample = None
        layers = []

        # 下采样
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers) # 建立Conv1_; Conv2_; Conv3_; Conv4..



class LitEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, act_fn = 'PReLU'):
        super(LitEncoder, self).__init__()
        self.latent_dim = latent_dim

        if act_fn == 'PReLU':
            self.act_fn = nn.PReLU()

        c_hid = 10
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, c_hid, kernel_size=3, padding = 1, stride = 2),
            self.act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding = 1),
            self.act_fn,
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            self.act_fn,
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            self.act_fn,
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride =2),
            self.act_fn,
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(15680, self.latent_dim)
        self.fc_log_var = nn.Linear(15680, self.latent_dim)



    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var



def ResNet50Encoder(latent_dim, channels=3, act_fn = 'PReLU'):
    return ResNetEncoder(Bottleneck, [3, 4, 6, 3], latent_dim, channels, act_fn=act_fn)

def ResNet101Encoder(latent_dim, channels = 3, act_fn='PReLU'):
    return ResNetEncoder(Bottleneck, [3, 4, 23, 3], latent_dim, channels, act_fn=act_fn)

def ResNet18Encoder(latent_dim, channels = 3, act_fn='PReLU'):
    return ResNetEncoder(Bottleneck, [2,2,2,2], latent_dim,  channels, act_fn=act_fn)

def LitEncoder1(latent_dim, channels = 3, act_fn ='PReLU'):
    return LitEncoder(latent_dim, channels, act_fn = act_fn)



if __name__ == '__main__':
     #net = ResNet18Encoder(latent_dim=512, channels=3, act_fn='PReLU')
     #x = torch.randn(2, 3, 224, 224)
     #mu, log_var = net(x)
     #print( mu.shape, log_var.shape)


     #net = LitEncoder1(latent_dim=512, act_fn='PReLU')
     #x = torch.randn(2,3,224,224)
     #mu, log_var = net(x)
     #print(mu.shape, log_var.shape)

     net = ResNet50Encoder(latent_dim=512, channels = 3, act_fn='PReLU')
     x = torch.randn(2,3,224,224)
     mu, log_var = net(x)
     print(mu.shape, log_var.shape)














