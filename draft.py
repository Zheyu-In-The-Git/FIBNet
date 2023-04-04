
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

'''
File "C:\Users\40398\PycharmProjects\Bottleneck_Nets\main.py", line 161, in <module>
    main(args)
  File "C:\Users\40398\PycharmProjects\Bottleneck_Nets\main.py", line 84, in main
    trainer.fit(model, datamodule=data_module)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 608, in fit
    call._call_and_handle_interrupt(
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\call.py", line 38, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 650, in _fit_impl
    self._run(model, ckpt_path=self.ckpt_path)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 1112, in _run
    results = self._run_stage()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 1191, in _run_stage
    self._run_train()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 1214, in _run_train
    self.fit_loop.run()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\fit_loop.py", line 267, in advance
    self._outputs = self.epoch_loop.run(self._data_fetcher)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\epoch\training_epoch_loop.py", line 213, in advance
    batch_output = self.batch_loop.run(kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\batch\training_batch_loop.py", line 88, in advance
    outputs = self.optimizer_loop.run(optimizers, kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 202, in advance
    result = self._run_optimization(kwargs, self._optimizers[self.optim_progress.optimizer_position])
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 249, in _run_optimization
    self._optimizer_step(optimizer, opt_idx, kwargs.get("batch_idx", 0), closure)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 370, in _optimizer_step
    self.trainer._call_lightning_module_hook(
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 1356, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\core\module.py", line 1742, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\core\optimizer.py", line 169, in step
    step_output = self._strategy.optimizer_step(self._optimizer, self._optimizer_idx, closure, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\strategies\strategy.py", line 234, in optimizer_step
    return self.precision_plugin.optimizer_step(
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\plugins\precision\precision_plugin.py", line 119, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\optim\optimizer.py", line 113, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\autograd\grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\optim\adam.py", line 118, in step
    loss = closure()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\plugins\precision\precision_plugin.py", line 105, in _wrap_closure
    closure_result = closure()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 149, in __call__
    self._result = self.closure(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 135, in closure
    step_output = self._step_fn()
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\loops\optimization\optimizer_loop.py", line 419, in _training_step
    training_step_output = self.trainer._call_strategy_hook("training_step", *kwargs.values())
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 1494, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\pytorch_lightning\strategies\strategy.py", line 378, in training_step
    return self.model.training_step(*args, **kwargs)
  File "C:\Users\40398\PycharmProjects\Bottleneck_Nets\model\bottleneck_nets.py", line 174, in training_step
    real_loss = self.configure_loss(real_z_discriminator_value, z_valid, 'BCE')
  File "C:\Users\40398\PycharmProjects\Bottleneck_Nets\model\bottleneck_nets.py", line 91, in configure_loss
    return bce(pred, true)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\loss.py", line 714, in forward
    return F.binary_cross_entropy_with_logits(input, target,
  File "C:\Users\40398\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\functional.py", line 3150, in binary_cross_entropy_with_logits
    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
'''
