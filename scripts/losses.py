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


# pixel-wise logistic regression with focal loss
# l1 losses for the two heads (wh,offset)


def mod_focal_loss(pred,gt):

	# print(preds.shape, gt.shape)

	pos_idx = gt.eq(1).float()
	neg_idx = gt.lt(1).float()
	
	neg_weights = torch.pow(1-gt, 4)


	# for pred in preds:
	pred = torch.clamp(pred, 1e-12)

	# pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_idx
	# print(torch.log(pred).shape)
	# print(torch.pow(1 - pred, 2).shape)

	pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_idx
	neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_idx

	num_pos = pos_idx.float().sum()
	pos_loss = pos_loss.sum()
	neg_loss = neg_loss.sum()

	if num_pos == 0 :
		loss = -neg_loss

	else:
		loss = -(pos_loss + neg_loss) / num_pos

	return loss
