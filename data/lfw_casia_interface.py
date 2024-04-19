import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .lfw_casia_data import LFWCasiaData, LFWCaisaRecognitionTestPairs

class LFWCasiaInterface(pl.LightningDataModule):
    def __init__(self,
                 dataset:str,
                 batch_size:int,
                 dim_img:int,
                 sensitive_attr:str,
                 lfw_data_dir:str,
                 casia_data_dir:str,
                 purpose:str):
        super(LFWCasiaInterface).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.sensitive_attr = sensitive_attr
        self.lfw_data_dir = lfw_data_dir
        self.casia_data_dir = casia_data_dir

        self.prepare_data_per_node = True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = True

        if purpose == 'attr_extract':
            self.trainset = LFWCasiaData(dim_img=self.dim_img,
                                         lfw_data_dir=self.lfw_data_dir,
                                         casia_data_dir=self.casia_data_dir,
                                         sensitive_attr=self.sensitive_attr,
                                         img_path_replace=True, split='all')
            self.testset = LFWCasiaData(dim_img=self.dim_img,
                                         lfw_data_dir=self.lfw_data_dir,
                                         casia_data_dir=self.casia_data_dir,
                                         sensitive_attr=self.sensitive_attr,
                                         img_path_replace=True, split='all')

        elif purpose == 'face_recognition':
            self.testset = LFWCaisaRecognitionTestPairs(dim_img=self.dim_img,
                                                        lfw_data_dir=self.lfw_data_dir,
                                                        casia_data_dir=self.casia_data_dir)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.trainset
        if stage == 'test' or stage is None:
            self.testset = self.testset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

if __name__ == '__main__':
    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    Male_dataloader = LFWCasiaInterface(dim_img=224,
                                        batch_size=100,
                                        dataset='lfw_casia_data',
                                        sensitive_attr='Male',
                                        lfw_data_dir=lfw_data_dir,
                                        casia_data_dir=casia_data_dir,
                                        purpose='attr_extract')
    Male_dataloader.setup(stage='test')

    for i, item in enumerate(Male_dataloader.test_dataloader()):
        print('i',i)
        x, u, s = item
        break




    White_dataloader = LFWCasiaInterface(dim_img=224,
                                        batch_size=100,
                                        dataset='lfw_casia_data',
                                        sensitive_attr='White',
                                        lfw_data_dir=lfw_data_dir,
                                        casia_data_dir=casia_data_dir,
                                        purpose='attr_extract')
    for i, item in enumerate(White_dataloader.test_dataloader()):
        print('i',i)
        x,u,s = item
        break


    Face_dataloader = LFWCasiaInterface(dim_img=224,
                                        batch_size=100,
                                        dataset='lfw_casia_data',
                                        sensitive_attr='White',
                                        lfw_data_dir=lfw_data_dir,
                                        casia_data_dir=casia_data_dir,
                                        purpose='face_recognition')
    for i, item in enumerate(Face_dataloader.test_dataloader()):
        print('i', i)
        img_x, img_y, match = item








