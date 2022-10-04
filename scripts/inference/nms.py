from torch.cuda import is_available
from torch._C import device
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random, tqdm, sys, math
import os
import cv2
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from PIL import ImageFile
import os
import timm
from torchsummary import summary
from losses import *
import torch.nn.functional as F
from dataset.coco import VOCDataset
from torch.utils.data import DataLoader


def nms(heatmap):
	heatmap_max_scores = F.max_pool2d(heatmap, 3, stride = 1, padding = 1)
	keep = (heatmap_max_scores == heatmap).float()
	return keep*heatmap


