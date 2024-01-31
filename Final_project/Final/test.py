# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:02:27 2023

@author: yushe
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

import torch
from torch import nn, optim
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from fastai.vision.models.unet import DynamicUnet

#import utils
#import model_def

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[]

style = "real"
size = 128
folder_path = os.getcwd() + '/test_data'
#model_save_path = os.getcwd() + '/model/G_' + style + '.pt'
model_save_path = r'C:\Users\EE803\Desktop\L1_1\model\NetG.pt'
transforms = T.Resize((size, size))

#net_G = model_def.build_res_unet(n_input=1, n_output=2, size=128)
net_G = torch.load(model_save_path, map_location=device)
net_G.eval()

for idx in range(12):
    img_path = folder_path + '/' + str(idx+1) + '.jpg'
    img = Image.open(img_path).convert("RGB")
    old_size = img.size
    img_name = os.path.basename(img_path).split('.jpg')[0]
    img = transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
    img_lab = T.ToTensor()(img_lab).to(device)
    real_L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    real_L = real_L.unsqueeze(0)
    with torch.no_grad():
        fake_ab = net_G(real_L)
    fake_img = utils.lab_to_rgb(real_L, fake_ab).squeeze(0)*255
    fake_img = fake_img.astype(np.uint8)
    fake_img = Image.fromarray(fake_img).resize(old_size)
    save_name = folder_path + '/' + img_name + '_' + style + '.jpg'
    fake_img.save(save_name)