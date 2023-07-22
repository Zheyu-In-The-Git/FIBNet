import matplotlib.backends.backend_pdf
import torch
import pytorch_lightning as pl
import os
import PIL

pl.seed_everything(83)

from matplotlib import pyplot as plt
from torchvision import transforms
# from arcface_resnet50 import ArcfaceResnet50
from sklearn.manifold import TSNE

from data import CelebaTSNEExperiment
from torch.utils.data import DataLoader
from model import BottleneckNets, Encoder, Decoder
import arcface_resnet50
import numpy as np
import pandas as pd
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # ******#


import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])
















