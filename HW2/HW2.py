import torch
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import numpy as np
import skimage as io
import os
import cv2
from PIL import Image
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
from functools import partial
from torch.optim import SGD
from random import sample
from IPython.display import display
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import argparse
#%%  load image
root_unlabel='C:/Users/EE803/Desktop/雲端運算與資料分析/HW2/toothdata/unlabeled'            #更改路徑
root_test='C:/Users/EE803/Desktop/雲端運算與資料分析/HW2/toothdata/test'                    #更改路徑

dataset_path = os.listdir(root_unlabel)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
#%%  prepare train data
image_path = []
# Add them to the list
for room in dataset_path:
    image_path.append(str(root_unlabel) + '/' + room)
#%%   prepare test data
test_dataset_path = os.listdir(root_test)
test_path=[]

for item in test_dataset_path:
 # Get all the file names
 names = os.listdir(root_test + '/' +item)
 for name in names:
     test_path.append(str(root_test + '/' + item)+'/'+name)


#%%   dataset     transform

h, w = 96,64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#color jitter
jitter_strength = 1
color_jitter = transforms.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength
        )
# gaussian blur
gaussian_blur = transforms.GaussianBlur(7,(2.0,8.0))




transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomApply([gaussian_blur], p=0.7),
    #transforms.RandomApply([color_jitter], p=0.8),
    #transforms.RandomResizedCrop((32, 32), scale=(0.6, 1.0)),
    transforms.Resize((h,w)),
    transforms.ToTensor(),   
    #transforms.Normalize(mean, std)
])

class Dataset(Dataset):

    def __init__(self, train_path, transform=None):
        self.train_path = train_path
        self.transform = transform        
        self.size=len(train_path)
        

    def __getitem__(self, idx):
        if self.transform is None:
            image_path = self.train_path[idx]
            image = Image.open(image_path)
            trf = transforms.ToTensor()
            xi=trf(image)
            return xi
        
        else:
            image_path = self.train_path[idx]
            image = Image.open(image_path)
            xi=transform(image)
            xj=transform(image)
            return xi , xj
    
    
    def __len__(self):
        return self.size



#%%  loss function
def xt_xent(
    u: torch.Tensor,                               # [N, C]
    v: torch.Tensor,                               # [N, C]
    temperature: float = 0.1):
    """
    N: batch size
    C: feature dimension
    """
    N,C= u.shape
    
    z = torch.cat([u, v], dim=0)                   # [2N, C]
    z = F.normalize(z, p=2, dim=1)                 # [2N, C]
    s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
    s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity
    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N)                            # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss
    return loss   

#%%    KNN  test
def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    print('=========embedding 的大小:   ',emb.shape)
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)


#%%  encoder
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
    #input size 3*224*224
        self.conv1=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
    #block2_1
        self.conv2_1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1=nn.BatchNorm2d(64)
        self.conv2_2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2=nn.BatchNorm2d(64)
    #block2_2
        self.conv2_3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_3=nn.BatchNorm2d(64)
        self.conv2_4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_4=nn.BatchNorm2d(64)
    #block3_1
        self.conv3_1=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1=nn.BatchNorm2d(128)
        self.conv3_2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2=nn.BatchNorm2d(128)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128))
    #block3_2
        self.conv3_3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_3=nn.BatchNorm2d(128)
        self.conv3_4=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_4=nn.BatchNorm2d(128)
    #block4_1
        self.conv4_1=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1=nn.BatchNorm2d(256)
        self.conv4_2=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2=nn.BatchNorm2d(256)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256))
    #block4_2
        self.conv4_3=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_3=nn.BatchNorm2d(256)
        self.conv4_4=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_4=nn.BatchNorm2d(256)

    #block5_1
        self.conv5_1=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1=nn.BatchNorm2d(512)
        self.conv5_2=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2=nn.BatchNorm2d(512)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512))
    #block5_2
        self.conv5_3=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_3=nn.BatchNorm2d(512)
        self.conv5_4=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_4=nn.BatchNorm2d(512)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc=nn.Linear(512,512)
        self.bn5_4=nn.BatchNorm2d(512)
        
        
    def forward(self, input):
        output=self.conv1(input)
        output=F.relu(self.bn1(output))
        output=self.pool1(output)
        #block2_1
        residul2_1=output
        output=self.conv2_1(output)
        output=F.relu(self.bn2_1(output))
        output=self.conv2_2(output)
        output=self.bn2_2(output)
        output+=residul2_1
        output=F.relu(output)
    #block2_2
        residul2_2=output
        output=self.conv2_3(output)
        output=F.relu(self.bn2_3(output))
        output=self.conv2_4(output)
        output=self.bn2_4(output)
        output+=residul2_2
        output=F.relu(output)
    #block3_1
        residul3_1=self.downsample1(output)
        output=self.conv3_1(output)
        output=F.relu(self.bn3_1(output))
        output=self.conv3_2(output)
        output=self.bn3_2(output)
        output+=residul3_1
        output=F.relu(output)
    #block3_2
        residul3_2=output
        output=self.conv3_3(output)
        output=F.relu(self.bn3_3(output))
        output=self.conv3_4(output)
        output=self.bn3_4(output)
        output+=residul3_2
        output=F.relu(output)

    #block4_1
        residul4_1=self.downsample2(output)
        output=self.conv4_1(output)
        output=F.relu(self.bn4_1(output))
        output=self.conv4_2(output)
        output=self.bn4_2(output)
        output+=residul4_1
        output=F.relu(output)
    #block4_2
        residul4_2=output
        output=self.conv4_3(output)
        output=F.relu(self.bn4_3(output))
        output=self.conv4_4(output)
        output=self.bn4_4(output)
        output+=residul4_2
        output=F.relu(output)

    #block5_1
        residul5_1=self.downsample3(output)
        output=self.conv5_1(output)
        output=F.relu(self.bn5_1(output))
        output=self.conv5_2(output)
        output=self.bn5_2(output)
        output+=residul5_1
        output=F.relu(output)
    #block3_2
        residul5_2=output
        output=self.conv5_3(output)
        output=F.relu(self.bn5_3(output))
        output=self.conv5_4(output)
        output=self.bn5_4(output)
        output+=residul5_2
        output=F.relu(output)

        output=self.avgpool(output)
        output = self.flatten(output)
        return output



#%%   train
def train(model,n_epochs,train_loader,test_loader,output_loader,classes,batch,optimizer,scheduler,criterion):
    train_total_acc=[]
    train_totale_losses=[]
    test_total_acc=[]
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        train_losses=[]
        loss_record=[]
        print('running epoch: {}'.format(epoch))
        print('=======training======')
        model.to(device)
        model.train()
        for xi, xj in tqdm(train_loader):
            if train_on_gpu:
                xi, xj = xi.to(device), xj.to(device)
            optimizer.zero_grad()
            hi = model(xi)
            hj = model(xj)
            #loss optimizer
            loss = criterion(hi, hj)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            
        train_loss = round(sum(loss_record)/len(loss_record),3)
        train_totale_losses.append(train_loss)
        scheduler.step(train_loss)
        #print(epoch, scheduler.get_last_lr()[0])
        print('\tTraining Loss: {:.6f} '.format(train_loss))
        ##save the model
        model_filename = '\model'+str(epoch)+'.pth'
        torch.save(model, r'C:\Users\EE803\Desktop\雲端運算與資料分析\HW2\model'+model_filename)   #儲存模型    #更改路徑
        
        #KNN    test
        print('=======KNN  testing======')
        model.eval()
        embedding = torch.tensor([])
        output_embedding=torch.tensor([])
        with torch.no_grad():
            for data in tqdm(test_loader):                                  #使用embedding 資料來測試準確率
                if train_on_gpu:
                    data,embedding = data.to(device),embedding.to(device)#丟到gpu上
                
                emb = model(data)
                embedding = torch.cat((embedding,emb), dim = 0)
            #embedding_tensor = torch.cat((embedding))
            acc = KNN(embedding, classes, batch_size=batch)
            #scheduler.step(acc)
            print("Accuracy: %.5f" % acc)
            test_total_acc.append(acc)
            
            #產生unlabel資料的np檔
            for xi in tqdm(output_loader):
                if train_on_gpu:
                    xi,output_embedding = xi.to(device),output_embedding.to(device)
                emb_test= model(xi)
                output_embedding = torch.cat((output_embedding,emb_test), dim = 0)
            output_embedding = output_embedding.cpu().numpy()     #產生的embedding轉成cpu tensor 再轉成numpy檔
            output_filename = '\output'+str(epoch)+'.npy'
            np.save(r'C:\Users\EE803\Desktop\雲端運算與資料分析\HW2\output'+output_filename,output_embedding)   #更改路徑
                
    return train_totale_losses,test_total_acc,model





#%%  dataloatder
#argements
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()



#model
model = ResNet18()

#batch_size=64
test_batch_size = 1
#train data
traindata=Dataset(image_path, transform)
train_loader = DataLoader(traindata, batch_size=args.batch_size,shuffle=True)
#test data
testdata=Dataset(test_path)
test_loader = DataLoader(testdata, batch_size=args.test_batch_size,shuffle=False)
#output data0
outputdata=Dataset(image_path)
output_loader = DataLoader(outputdata, batch_size=test_batch_size,shuffle=False)
# summary writer
#train_writer = SummaryWriter(r'C:\Users\EE803\Desktop\HW2\train_writer')



N = 100
#optimizer = torch.optim.Adam(model.parameters(), lr=Learingrate)
optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=3, verbose=True)

classes = torch.cat([
    torch.zeros(N),
    torch.ones(N),
    #torch.ones(N)*2,
    #torch.ones(N)*3
], dim=0)

train_losses,test_accurancy, model = train(model,args.epochs,train_loader,test_loader,output_loader,classes,args.batch_size,optimizer,train_scheduler,xt_xent)

print ("the maximum accurancy index is:  ",test_accurancy.index(max(test_accurancy)))
print("the maximum accurancy is :   ",max(test_accurancy))


# plot

x = np.linspace(0, args.epochs-1, args.epochs)
plt.xlabel('epoh', fontsize = 16)                        # 設定坐標軸標籤
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
# 繪圖並設定線條顏色、寬度、圖例
line1, = plt.plot(x, test_accurancy, color = 'red', linewidth = 3, label = 'test accurancy')             
plt.legend(handles = [line1], loc='upper right')
plt.show()
line3, = plt.plot(x, train_losses, color = 'blue', linewidth = 3, label = 'train loss')
plt.legend(handles = [line3], loc='upper right')
plt.show()










