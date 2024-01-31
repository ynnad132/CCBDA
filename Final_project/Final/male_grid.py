import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms 
from torchvision.utils import make_grid
from torchvision.utils import save_image


path = os.getcwd() +'/griddata/gray'

paths =glob.glob(os.path.join(path, "*")) # Grabbing all the image file names


Tensor = torch.cuda.FloatTensor
image_tensor = torch.Tensor([])

transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()]) 

for i in paths:
    img = Image.open(i)
    img = transform(img).unsqueeze(0)
    image_tensor = torch.cat((image_tensor,img),0)
    
    
save_image(image_tensor, 'gray_griddata.png',nrow = 4)


