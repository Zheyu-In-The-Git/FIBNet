import torch
import torch.nn as nn
import math

class UtilityDiscriminator(nn.Module):
    def __init__(self, utility_dim):
        super(UtilityDiscriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(utility_dim, 3000),
            nn.BatchNorm1d(3000),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(3000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 1),
            #nn.Sigmoid() #TODO：在UtilityDiscriminator模型中，因为在模型训练阶段，要求用nn.BCEWithLogitsLoss()
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
    net = UtilityDiscriminator(utility_dim=2622)
    x = torch.randn(2, 2622)
    # print(x.shape)

    # print(net)
    out = net(x)
    print(out)

