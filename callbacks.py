import pytorch_lightning as pl
import torch


class GenerateCallback(pl.callbacks):
    def __init__(self, input_imgs, every_n_epoch = 1):
        super(GenerateCallback, self).__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            pass
