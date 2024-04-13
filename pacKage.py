import os, sys
import numpy as np
from random import seed, shuffle
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torchsummary import summary
device = "cuda" if torch.cuda.is_available() else "cpu"
import cv2
import glob
from torch.utils.data import DataLoader, Dataset
from typing import *
from pathlib import Path
from glob import glob
import tifffile as tiff
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize #what the heck is this, so useful
from random import seed, shuffle
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import cv2
import importlib
from tqdm import *