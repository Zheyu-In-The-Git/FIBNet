import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from data.celeba_data import CelebaData
import pickle
import copyreg
import io



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
        #
        # 数据加载
        # self.load_data_module()
    #
    def prepare_data(self):
         CelebaData(dim_img=self.dim_img, data_dir=self.data_dir, sensitive_dim=self.sensitive_dim,identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr, split='all')

    def setup(self, stage=None):

        if self.dataset == 'celeba_data':
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                self.trainset = CelebaData(dim_img=self.dim_img, data_dir=self.data_dir, sensitive_dim=self.sensitive_dim,
                                            identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr, split='train')

                self.valset = CelebaData(dim_img=self.dim_img, data_dir=self.data_dir, sensitive_dim=self.sensitive_dim,
                                            identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr, split='valid')
            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testset = CelebaData(dim_img=self.dim_img, data_dir=self.data_dir, sensitive_dim=self.sensitive_dim,
                                            identity_nums=self.identity_nums, sensitive_attr=self.sensitive_attr, split='test')

            # # If you need to balance your data using Pytorch Sampler,
            # # please uncomment the following lines.

            # with open(self.data_dir + '/samples_weight.pkl', 'rb') as f:
            #     self.sample_weight = pkl.load(f)

    def train_dataloader(self):
        # sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset) * 20)
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)

'''



    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
            print(self.data_module)
        except:
            raise ValueError('Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')


    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

'''

if __name__ == '__main__':
    data_dir = 'D:\celeba' # 'D:\datasets\celeba'
    dataloader = CelebaInterface(dim_img=224,dataset='celeba_data', data_dir=data_dir, sensitive_dim=2, identity_nums=10177, sensitive_attr='Male', batch_size=10, num_workers=0, pin_memory=False)
    dataloader.setup(stage='fit')

    for i, item in enumerate(dataloader.val_dataloader()):
        print('i', i )
        x, u, s = item
        print('u', u)
        break


