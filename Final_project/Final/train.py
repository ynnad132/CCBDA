# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:48:01 2023

@author: yushe
"""

# In[]
import os
import glob
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from fastai.vision.models.unet import DynamicUnet

import utils
import model_def


# In[]

current_path = os.getcwd()  # "C:/Users/yushe/Documents/CCBDA/Final"
config = {
    'seed': 2023,
    'img_size': 128,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# np.random.seed(config['seed'])
os.makedirs(os.getcwd()+'/model', exist_ok=True)

# In[]: Pretrain Generator

def pretrain_generator(epochs, device='cpu', style="real"):
    train_dl, _ = utils.get_dataloader(style=style)
    net_G = model_def.build_res_unet(n_input=1, n_output=2, size=128)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss() 
    
    for e in range(epochs):
        loss_meter = utils.AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
    
    save_path = os.getcwd() + '/model/pretrained_' + style + '.pt'
    torch.save(net_G.state_dict(), save_path)
    return save_path
       
# In[]

def train_model(pretrained_model_path, epochs=20, display_every=200, style="real"):
    train_dl, val_dl = utils.get_dataloader(style=style)
    val_data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    net_G = model_def.build_res_unet(n_input=1, n_output=2, size=128)
    net_G.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model = model_def.MainModel(net_G=net_G)

    for e in range(epochs):
        loss_meter_dict = utils.create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            utils.update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                utils.log_results(loss_meter_dict) # function to print out the losses
                utils.visualize(model, val_data, save=False) # function displaying the model's outputs
            
    save_path = os.getcwd() + '/model/G_' + style + '.pt'
    torch.save(net_G.state_dict(), save_path)

# In[]: Style = Real

model_path = pretrain_generator(20, device=device, style="real")

# In[]: Style = Real

train_model(model_path, 20, style="real")

# In[]: Style = style1

model_path = pretrain_generator(20, device=device, style="style1")

# In[]: Style = style1
    
train_model(model_path, 20, style="style1")

# In[]: Style = style2

model_path = pretrain_generator(20, device=device, style="style2")

# In[]: Style = style2
    
train_model(model_path, 20, style="style2")

