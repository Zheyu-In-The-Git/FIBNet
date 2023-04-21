
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data.celeba_data import CelebaData, CelebaRecognitionTestDataSet
import torch.utils.data as data


class CelebaInterface(pl.LightningDataModule):

    def __init__(self,
                 num_workers : int,
                 dataset: str,
                 batch_size : int,
                 dim_img : int,
                 data_dir : str,
                 sensitive_dim : int,
                 identity_nums : int,
                 sensitive_attr : str,
                 **kargs
                 ):
        super(CelebaInterface).__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = batch_size

        # 数据集相关的参数
        self.dim_img = dim_img
        self.data_dir = data_dir
        self.sensitive_dim = sensitive_dim
        self.identity_nums = identity_nums
        self.sensitive_attr = sensitive_attr
        self.pin_memory = kargs['pin_memory']

        self.prepare_data_per_node = True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices =True

    def prepare_data(self):
         CelebaData(dim_img=self.dim_img, data_dir=self.data_dir, sensitive_dim=self.sensitive_dim,identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr, split='all')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_valid_dataset = CelebaData(dim_img=self.dim_img,
                                             data_dir=self.data_dir,
                                             sensitive_dim=self.sensitive_dim,
                                             identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr,
                                             split='train_valid_70%')
            seed = torch.Generator().manual_seed(83)
            valid_set_size = int(len(train_valid_dataset) * 0.1)
            train_set_size = len(train_valid_dataset) - valid_set_size

            self.Train_Dataset, self.Valid_Dataset = data.random_split(train_valid_dataset,[train_set_size, valid_set_size], generator=seed)

        if stage == 'test' or stage is None:
            self.Test_Dataset = CelebaRecognitionTestDataSet(dim_img=self.dim_img, data_dir=self.data_dir)


    def train_dataloader(self):
        # sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset) * 20)
        return DataLoader(self.Train_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.Valid_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.Test_Dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)


if __name__ == '__main__':
    # data_dir = 'D:\celeba' # 'D:\datasets\celeba'
    data_dir = '/Volumes/xiaozhe_SSD/datasets/celeba'
    dataloader = CelebaInterface(dim_img=224,dataset='celeba_data', data_dir=data_dir, sensitive_dim=2, identity_nums=10177, sensitive_attr='Male', batch_size=2, num_workers=0, pin_memory=False)
    dataloader.setup(stage='test')

    for i, item in enumerate(dataloader.test_dataloader()):
        print('i', i )
        img_x, img_y, match = item
        print(img_x)
        print(img_y)
        print(match)
        break



