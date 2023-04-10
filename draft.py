
import torch
import pytorch_lightning as pl
pl.seed_everything(43)
from model import utility_discriminator
import torch.nn as nn
'''
# 二元交叉熵
x = torch.Tensor([0.1, 0.5, 0.4])
y = torch.Tensor([0,1,0])

bce = nn.BCELoss()
dis_bce = bce(x,y)
print(dis_bce)
total = -torch.log(torch.Tensor([0.5])) -torch.log(torch.Tensor([0.9])) -torch.log(torch.Tensor([0.6]))
print(total/3)


x = torch.Tensor([1, 2, 3])
sigmoid = nn.Sigmoid()
print(sigmoid(x))
'''
'''
# 写一下detach()

a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
a = a.detach() # 会将requires_grad 属性设置为False
print(a.requires_grad)
'''
'''

# 考察一下torchmetrics
import torchmetrics
preds = torch.randn(10, 5).softmax(dim=-1)
print(preds.shape)
target = torch.randint(5, (10,))
print(target.shape)

acc = torchmetrics.functional.accuracy(preds, target)
print(acc)
'''
# import torch
# print(torch.__version__)



# a = torch.tensor([1., 2., 3.], requires_grad = True)
# b = a.clone()
#
# print(a.data_ptr())
# print(b.data_ptr())
#
# print(a)
# print(b)
# print('-'*30)
#
# c = a * 2
# d = b * 3
#
# c.sum().backward()
# print(a.grad)
#
# d.sum().backward()
# print(a.grad)
# print(b.grad)
#
# print('-'*60)
'''


a = torch.tensor([1., 2., 3.],requires_grad=True)
b = a.detach()

print(a.data_ptr()) # 2432102290752
print(b.data_ptr()) # 2432102290752 # 内存位置相同

print(a) # tensor([1., 2., 3.], requires_grad=True)
print(b) # tensor([1., 2., 3.]) # 这里为False，就省略了
print('-!'*30)

c = a * 2
d = b * 3

c.sum().backward()
print(a.grad) # tensor([2., 2., 2.])

# d.sum().backward()
print(a.grad) # 报错了！ 由于b不记录计算图，因此无法计算b的相关梯度信息
print(b.grad)

print( 'gpu count: ',torch.cuda.device_count())


print(pl.__version__)
'''



'''
# 审查模型参数
pretrained_filename = 'lightning_logs/bottleneck_test_version_1/checkpoints/saved_models/epoch=0-step=400.ckpt'
bottlenecknets = ConstructBottleneckNets(args)
model = bottlenecknets.load_from_checkpoint(pretrained_filename)

x = torch.randn(2, 3, 224, 224)
z, u_hat, s_hat, u_value, s_value, mu, log_var = model(x)
print(z, mu, log_var)
'''
'''
sigmoid = nn.Sigmoid()
def kl_estimate_value(discriminating):
    discriminated = sigmoid(discriminating)
    kl_estimate_value = (torch.log(discriminated) - torch.log(1 - discriminated)).sum(1).mean()
    return kl_estimate_value.detach()


net = utility_discriminator.UtilityDiscriminator(utility_dim=10177)
x = torch.randn(3, 10177)
out = net(x)
print(out)

value = kl_estimate_value(out)
print(value)
'''


