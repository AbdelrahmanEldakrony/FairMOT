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
import torch.nn.functional as F
from dataset.coco import VOCDataset
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageDraw
from torchvision.io import read_image
from loss.loss import Loss


class Head(nn.Module):

	def __init__(self, num_classes=1,channels=256):
		super(Head, self).__init__()

		self.model_dla34 = timm.create_model('dla34', pretrained=True, features_only=True)
		# print(f'Feature channels: {self.model_dla34.feature_info.channels()}')
		# print(f'Feature channels: {self.model_dla34.feature_info.reduction()}')
		# exit()
		self.hm_head = nn.Sequential(nn.Conv2d(64, channels,3, padding = 1, bias=True),
			# nn.BatchNorm2d(channels),
			nn.ReLU(),
			nn.Conv2d(channels, num_classes, 1, stride=1, padding = 0))

		self.box_size_head = nn.Sequential(

			nn.Conv2d(64, channels, 3, padding=1),
			# nn.BatchNorm2d(channels),
			nn.ReLU(),
			nn.Conv2d(channels, 2, 1)
			)

		self.box_offset_head = nn.Sequential(

			nn.Conv2d(64, channels, 3, padding=1),
			# nn.BatchNorm2d(channels),
			nn.ReLU(),
			nn.Conv2d(channels, 2, 1)
			)

	def forward(self, x):
		x = self.model_dla34(x)[1]
		hm = self.hm_head(x).sigmoid()
		# print('HEAT:',hm.shape)
		wh = self.box_size_head(x).sigmoid()
		offset = self.box_offset_head(x).sigmoid()

		return hm, wh, offset



def train():
	train_ds = VOCDataset()
	print(train_ds.category2id)
	# train_size = int(1 * len(train_ds))
	# test_size = len(train_ds) - train_size
	# print(train_size)
	# print(test_size)
	# train_split, test_split = torch.utils.data.random_split(train_ds, [train_size,test_size], generator=torch.Generator().manual_seed(42))
	train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, collate_fn=train_ds.collate_fn, pin_memory=True)
	# for j in train_dl:
	# 	img, boxes, classes, hm, info = j
	# 	print('Image shape:3', img.shape)
	# 	img = img[0]
	# 	print(type(img))
	# 	img= torchvision.transforms.ToPILImage()(img)

	# 	img.show()
	# 	exit()
	# 	print('Imagem shaape4:', img.size)
	# 	draw = ImageDraw.Draw(img)
	# 	print(boxes.shape)
	# 	boxes = boxes[0].tolist()
	# 	plt.figure(figsize=(10, 10))
	# 	print(boxes)
	# 	for b in boxes:
	# 		draw.rectangle(xy=b, outline='red', width=3)
	# 	plt.imshow(img)
	# 	plt.show()
	# 	break
	# exit()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	epochs = 30

	model = Head(num_classes=1)
	# model_name = "person_ep30_lr0001_sch0.1_10s_bt8_nw4"
	# model.load_state_dict(torch.load('../model/'+model_name+'.pt'))
	if torch.cuda.is_available():
		model = model.cuda()

	losser = Loss()
	losser = losser.cuda()
	alpha = 1.
	beta = 0.1
	gamma = 1.

	learning_rate = 0.001
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.3, verbose = True)
	# An image in a batch consists of: img, boxes, classes, hm/4, info/4
	# plt.imshow(i[0].squeeze().permute(1, 2, 0))

	best_loss = 10000000.
	for epoch in range(epochs):
		model.train(True)
		train_loss = 0
		cls_loss = 0.
		reg_loss = 0.
		step = 0
		for (gt) in tqdm.tqdm(train_dl):
			if model.cuda():
				gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

			optimizer.zero_grad()
			# print(gt[0].shape)
			pred = model(gt[0])
			losses = losser(pred, gt)
			cls_loss, reg_loss = losses
			loss = sum(losses)
			loss.backward()
			optimizer.step()
			train_loss+=loss.item()
			step+=1

		print('cls loss:', cls_loss.item(), 'reg loss:', reg_loss.item())
		print(f"Loss after {epoch} epoch(s):", train_loss)
		if train_loss == np.nan:
			break

		# if train_loss < best_loss:
		print('saving model...')
			# best_loss = train_loss
		torch.save(model.state_dict(), '../model/person_ep30_lr001_sch0.5_10s_bt8_nw4.pt')
		lr_scheduler.step()

		


if __name__ == '__main__':
	train()