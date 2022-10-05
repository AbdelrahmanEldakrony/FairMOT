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
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageDraw
from torchvision.io import read_image
import sys
sys.path.append("../")
import script as sc
from nms import nms
import numpy as np

# CLASSES_NAME = (
#         'aeroplane', 'bicycle', 'bird', 'boat',
#         'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#         'diningtable', 'dog', 'horse', 'motorbike',
#         'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')



CLASSES_NAME = ('person')


def preprocess_img(img, input_ksize):

    h, w = img.height, img.width

    img_resized = np.array(img.resize((512, 512)))

    # img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}


names = ['RTR2LP34edit','13cameras10-articleLarge','Jaywalking-Frames-Closeup','2009_002297','16150947010_78ed06b4e0_b','img2']

model_name = "person_ep62_lr0001_sch0.5_10s_bt8_nw4"
model = sc.Head(num_classes=1)
model.load_state_dict(torch.load('../../model/'+model_name+'.pt'), strict = False)
model.eval()


def test_image(img_name = None, img = None, mode = None):

    if mode != 'video':
        img = Image.open(f'/home/abdelrahman/Team/FairMOT/test_imgs/{img_name}.jpg').convert('RGB')

    img_mod, info = preprocess_img(img, [512,512])
    img_tensor = transforms.ToTensor()(img_mod)
    img_tensor = transforms.Normalize(std=[0.40789654, 0.44719302, 0.47026115], mean=[0.28863828, 0.27408164, 0.27809835])(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    _, _, h, w = img_tensor.shape
    pred_hm, pred_wh, pred_off = model(img_tensor)
    # pred_wh = pred_wh*512

    pred_hm = nms(pred_hm)
    b, c, output_h, output_w = pred_hm.shape

    print(h,w)
    print(torch.max(pred_wh))
    print(pred_hm.shape)
    # exit()
    centers = []
    box_size = []
    offset = []
    clses = []
    trsh = 0.08
    res = 0

    for i in range(128):
        for j in range(128):
            val = pred_hm[0,0,i,j]
            val = val.item()
            if(val<trsh):
                continue

            # centers.append((i,j))
            centers.append((i+pred_off[0,1,i,j].item(),j+pred_off[0,0,i,j].item()))
            # centers.append((i,j))
            box_size.append((pred_wh[0,0,i,j].item(),pred_wh[0,1,i,j].item()))
            offset.append((pred_off[0,0,i,j].item(),pred_off[0,1,i,j].item()))
            # clses.append(ind)



    img_mod = Image.fromarray(np.uint8(img_mod)).convert('RGB')

    draw = ImageDraw.Draw(img_mod)
    # box = [44,117,147,265]
    # draw.rectangle(xy=box, outline='red', width=3)
    for i in range(len(centers)):

        # xmin = ( centers[i][1] * (w / output_w) )- ( (box_size[i][0]*512)/2) 
        xmin = ( centers[i][1] - (box_size[i][0]* 128)/2 ) * (w / output_w) 
        
        # xmax = (centers[i][1] * (w / output_w) )+ ((box_size[i][0]*512)/2)
        xmax = ( centers[i][1] + (box_size[i][0]*128)/2) * (w / output_w)
        
        # ymin = (centers[i][0] * (h / output_h) )- ((box_size[i][1]*512)/2 )
        ymin = ( centers[i][0]  - (box_size[i][1]* 128)/2 ) * (h / output_h)

        # ymax = (centers[i][0] * (h / output_h) )+((box_size[i][1]*512)/2 )
        ymax = ( centers[i][0]  + (box_size[i][1]* 128)/2 ) * (h / output_h)


        box = [xmin, ymin, xmax, ymax]
        # box = torch.tensor(box)
        # box = box.unsqueeze(0)
        draw.rectangle(xy=box, outline='red', width=3)

    # plt.figure(figsize=(10, 10))
    # for i in range(len(centers)):
    #     # print(clses[i])
    #     plt.text(x=centers[i][1] * (w / output_w) - (box_size[i][0]*512)/2 ,y=centers[i][0] * (h / output_h) - (box_size[i][1]*512)/2 , s='Person', wrap=True, size=15,
    #                  bbox=dict(facecolor="r", alpha=0.7))
    print(img_mod.size)

    if mode == 'video':
        return img_mod
    else:
        plt.imshow(img_mod)
        plt.show()


def test_video(name):

    VIDEO_SOURCE = f'/home/abdelrahman/Team/FairMOT/test_imgs/{name}'
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    cnt=0
    while video_capture.isOpened():

        success, frame = video_capture.read()

        if not success:
            break
        
        # if cnt%5 != 0 :
        #     cnt+=1
        #     continue

        img = Image.fromarray(np.uint8(frame)).convert('RGB')
    
        img = test_image(img = img, mode='video')
        
        open_cv_img = np.array(img) 
        open_cv_img = open_cv_img[:, :, ::-1].copy() 
        cv2.imwrite('/home/abdelrahman/Team/FairMOT/test_imgs/frames/frame'+str(cnt)+'.jpg', open_cv_img)
        cnt+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def write_video():
    import glob
    img_array = []
    names = []
    size = (0,0)
    for filename in glob.glob('/home/abdelrahman/Team/FairMOT/test_imgs/frames/*.jpg'):
        names.append(filename)

    names.sort()
    print(len(names))
    for file in range(len(names)):
        img = cv2.imread(f'/home/abdelrahman/Team/FairMOT/test_imgs/frames/frame{file}.jpg')
        height, width, layers = img.shape

        size = (width,height)
        img_array.append(img)
     
     
    out = cv2.VideoWriter('/home/abdelrahman/Team/FairMOT/test_imgs/out2.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, size) 

    # print(img_array)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


for i in names:
    test_image(img_name=i)
# test_video('190625_04_CityMisc_HD_05.mp4')

# write_video()

# MP4V