
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.lfw_data import LFWData, LFWRecognitionTestPairs

import platform

class LFWInterface(pl.LightningDataModule):
    def __init__(self,
                 num_workers: int,
                 dataset:str,
                 batch_size:int,
                 dim_img:int,
                 sensitive_dim:int,
                 identity_nums:int,
                 sensitive_attr:str,
                 data_dir:str,
                 purpose:str,
                 **kwargs):
        super(LFWInterface).__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dir = data_dir

        # 数据集相关参数
        self.dim_img = dim_img
        self.dataset = dataset
        self.sensitive_dim = sensitive_dim
        self.identity = identity_nums
        self.sensitive_attr = sensitive_attr
        self.pin_memory = kwargs['pin_memory']


        self.prepare_data_per_node =True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = True

        system_platform = platform.platform().lower()

        if "darwin-22.1.0-x86_64-i386-64bit" in system_platform:
            self.LFWData_img_path_replace = True # mac 处理lfw 把图像路径\\ -> /
            self.LFWRecognition_img_path_replace = False # mac 不需要处理
        else:
            self.LFWData_img_path_replace = False # windows 不需要处理 直接用\\
            self.LFWRecognition_img_path_replace = True # windows 要把图像路径/ -> \\

        if purpose == 'attr_extract':
            self.trainset = LFWData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=self.identity,
                                    sensitive_attr=self.sensitive_attr, img_path_replace=self.LFWData_img_path_replace,
                                    split='train')
            self.testset = LFWData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=self.identity,
                                    sensitive_attr=self.sensitive_attr, img_path_replace=self.LFWData_img_path_replace,
                                    split='test')

        elif purpose == 'face_recognition':
            self.trainset = LFWData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=self.identity,
                                    sensitive_attr=self.sensitive_attr, img_path_replace=self.LFWData_img_path_replace,
                                    split='train')
            self.testset = LFWRecognitionTestPairs(dim_img=self.dim_img, data_dir=self.data_dir,
                                                   img_path_replace=self.LFWRecognition_img_path_replace)

    def prepare_data(self):
        LFWData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=self.identity, sensitive_attr=self.sensitive_attr, img_path_replace=self.LFWData_img_path_replace, split='train')
        LFWData(dim_img=self.dim_img, data_dir=self.data_dir, identity_nums=self.identity, sensitive_attr=self.sensitive_attr, img_path_replace=self.LFWData_img_path_replace, split='test')
        LFWRecognitionTestPairs(dim_img=self.dim_img, data_dir=self.data_dir, img_path_replace=self.LFWRecognition_img_path_replace)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.trainset
        if stage == 'test' or stage is None:
            self.testset = self.testset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)


if __name__ == '__main__':
    data_dir = '/Volumes/xiaozhe_SSD/datasets/lfw/lfw112'
    dataloader = LFWInterface(dim_img=224, dataset='LFW_data', data_dir=data_dir, sensitive_attr='Male', batch_size=2, num_workers=0, pin_memory=False, identity_nums=5749, sensitive_dim=1, purpose='attr_extract')
    dataloader.setup(stage='test')

    for i, item in enumerate(dataloader.test_dataloader()):
        print('i', i)
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break



