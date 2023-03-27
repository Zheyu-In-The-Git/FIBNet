import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SensitiveDiscriminator(nn.Module):
    def __init__(self, sensitive_dim):
        super(SensitiveDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(sensitive_dim, 4*sensitive_dim),
            nn.BatchNorm1d(4*sensitive_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(4*sensitive_dim, 1),
            #nn.Sigmoid() #TODO：在SensitiveDiscriminator模型中，因为在模型训练阶段，要求用nn.BCEWithLogitsLoss()
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
    net = SensitiveDiscriminator(sensitive_dim = 2)
    x = torch.randn(3, 2)
    # print(x.shape)

    # print(net)
    out = net(x)
    print(out.shape)
