import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        Conv_BN_IN_LReLU = []
        Conv_BN_IN_LReLU.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        Conv_BN_IN_LReLU.append(nn.BatchNorm2d(out_channels))
        Conv_BN_IN_LReLU.append(nn.InstanceNorm2d(out_channels))
        Conv_BN_IN_LReLU.append(nn.LeakyReLU())
        self.conv_bn_in_leakyrelu = nn.Sequential(*Conv_BN_IN_LReLU)

    def forward(self, x):
        out = self.conv_bn_in_leakyrelu(x)
        return out

class ConvTransporseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransporseBlock, self).__init__()
        ConvTransporse_BN_IN_LReLU = []
        ConvTransporse_BN_IN_LReLU.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        ConvTransporse_BN_IN_LReLU.append(nn.BatchNorm2d(out_channels))
        ConvTransporse_BN_IN_LReLU.append(nn.InstanceNorm2d(out_channels))
        ConvTransporse_BN_IN_LReLU.append(nn.LeakyReLU())
        self.convtransporse_bn_in_leakyrelu = nn.Sequential(*ConvTransporse_BN_IN_LReLU)

    def forward(self, x):
        out = self.convtransporse_bn_in_leakyrelu(x)
        return out



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.left_conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.left_conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.left_conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.left_conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.left_conv5 = ConvBlock(in_channels=512, out_channels=1024)

        self.right_conv5 = ConvTransporseBlock(in_channels=1024+40, out_channels=1024-512)
        self.right_conv4 = ConvTransporseBlock(in_channels=1024, out_channels=512 - 256)
        self.right_conv3 = ConvTransporseBlock(in_channels=512, out_channels=256)
        self.right_conv2 = ConvTransporseBlock(in_channels=256, out_channels=128)

        self.right_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x, s):
        x = self.left_conv1(x)
        x = self.left_conv2(x)
        x = self.left_conv3(x)
        x3_identity = x
        x = self.left_conv4(x)
        x4_identity = x
        x = self.left_conv5(x)

        s = s.view(s.size(0), s.size(1), 1, 1)
        s = s.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x,s], dim=1)

        x = self.right_conv5(x)
        x = torch.cat((x, x4_identity), dim=1)
        x = self.right_conv4(x)
        x = torch.cat((x, x3_identity), dim=1)
        x = self.right_conv3(x)
        x = self.right_conv2(x)
        x = self.right_conv1(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv5 = ConvBlock(in_channels=512, out_channels=1024)
        self.fc1_1 = nn.Linear(16, 1)
        self.fc1_2 = nn.Linear(1024,1)
        self.fc2_1 = nn.Linear(16, 1)
        self.fc2_2 = nn.Linear(1024, 40)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), x.size(1), 4*4)

        real = self.fc1_1(x)
        real = real.view(real.size(0),1, real.size(1) * real.size(2)) # 不行的话这里改一下
        real = self.fc1_2(real)
        real = self.sigmoid(real) # 到时候用 torch.nn.BCELoss()看下可不可以
        print(real)

        sensitive_attribute = self.fc2_1(x)
        sensitive_attribute = sensitive_attribute.view(sensitive_attribute.size(0), 1, sensitive_attribute.size(1) * sensitive_attribute.size(2))
        sensitive_attribute = self.fc2_2(sensitive_attribute)


        return real, sensitive_attribute









if __name__ == '__main__':
    x = torch.rand(size=(8, 3, 128, 128))
    s = torch.rand(size=(8, 40))
    generator = Generator()
    out = generator(x, s)
    print(out.size())

    discriminator = Discriminator()
    out = discriminator(out)





