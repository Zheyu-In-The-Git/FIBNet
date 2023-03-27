import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.optim as optim
import os
import math
from data import CelebaInterface
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

# 要不要在这里写一下resnet50呢
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

        super(ResNetEncoder, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)

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

        self.fc = nn.Linear(512* ResBlock.expansion, latent_dim)

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
        x = self.act_fn(self.batch_norm1(self.conv1(x)))

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

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


def ResNet50Encoder(latent_dim, channels=3, act_fn = 'ReLU'):
    return ResNetEncoder(Bottleneck, [3, 4, 6, 3], latent_dim, channels, act_fn=act_fn)







class FaceRecognizer(pl.LightningModule):
    def __init__(self):
        super(FaceRecognizer, self).__init__()
        self.resnet50 = ResNet50Encoder(512, channels=3, act_fn='ReLU')
        self.classifier = nn.Linear(512, 10177)
        #self.save_hyperparameters()

    def forward(self, x):
        x = self.resnet50(x)
        z = x
        u = self.classifier(x)
        return u, z

    def training_step(self, batch, batch_idx):
        x, u, _ = batch
        u_hat, _ = self(x)
        loss = F.cross_entropy(u_hat, u)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u , _ = batch

        preds = self.foward(x).argmax(dim=-1)
        acc = (u == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, u, _ = batch
        preds = self.forward(x).argmax(dim=-1)
        acc = (u == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)


    def configure_optimizers(self):
        b1 = 0.5
        b2 = 0.999
        optim_train = optim.Adam(self.parameters(), lr=0.001, betas=(b1, b2))
        return optim_train






CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/face_recognizer_resnet50/checkpoints/')
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    data_module = CelebaInterface(**vars(args))

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH + '/lightning_log', name='tensorboard_log',
                               version='version_1', )  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        # How many epochs to train for if no patience is set
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = FaceRecognizer.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(43)  # To be reproducable
        model = FaceRecognizer(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = FaceRecognizer.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    return model, result


if __name__ == '__main__':
    x = torch.randn(1, 3, 112,112)
    net = FaceRecognizer()
    output, latent = net(x)
    print(output.shape, latent.shape)





