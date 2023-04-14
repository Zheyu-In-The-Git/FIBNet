
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from data.celeba_data import CelebaData, CelebaRecognitionValidationDataSet, CelebaRecognitionTestDataSet
import pickle
import copyreg
import io
from data.lfw_data import LFWData, LFWRecognitionTestPairs

class LFWInterface(pl.LightningDataModule):
    pass