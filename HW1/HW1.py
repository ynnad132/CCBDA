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
from resnet18 import ResNet18
from random import sample
#%%  load video
root_train='C:/Users/EE803/Desktop/HW1/train'
root_test='C:/Users/EE803/Desktop/HW1/test'
dataset_path = os.listdir(root_train)

label_types = os.listdir(root_train)
print (label_types)  

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
#%%  prepare train data

rooms = []
for item in dataset_path:
 # Get all the file names
 all_rooms = os.listdir(root_train + '/' +item)

 # Add them to the list
 for room in all_rooms:
    rooms.append((item, str(root_train + '/' +item) + '/' + room))
    
# Build a dataframe        
train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(train_df.head())
print(train_df.tail())

df = train_df.loc[:,['video_name','tag']]
df
df.to_csv('train.csv')


#%%  準備tset資料

testset_path = os.listdir(root_test)
rooms1 = []

for item in testset_path:
 # Get all the file names
 rooms1.append( str(root_test+'/'+item ))


# Build a dataframe        
test_df = pd.DataFrame(data=rooms1, columns=['video_name'])
print(test_df.head())
print(test_df.tail())

dftest = test_df.loc[:,['video_name']]
dftest
dftest.to_csv('test.csv')


#%%    分資料

train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")

train_list=train['video_name'].tolist()
train_tag_list=train['tag'].tolist()

test_list=test['video_name'].tolist()


print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")


trainAddtag=[]
for i in range(len(train_list)):
    trainAddtag.append([train_list[i],train_tag_list[i]]) 
    
trainAddtag_val=sample(trainAddtag,2000)

for j in range(len(trainAddtag_val)):
    if trainAddtag_val[j] in trainAddtag:
        trainAddtag.remove(trainAddtag_val[j])
  
#%%

train_true=[]
train_tag_true=[]
val_true=[]
val_tag_true=[]


for k in range(len(trainAddtag)):
    train_true.append(trainAddtag[k][0])
    train_tag_true.append(trainAddtag[k][1])
    

for l in range(len(trainAddtag_val)):
    val_true.append(trainAddtag_val[l][0])
    val_tag_true.append(trainAddtag_val[l][1])



#%%取影片的frame                  等間隔取固定張數   
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames, dtype=np.int16)
    if (v_len>=n_frames):
        for fn in range(v_len):
            success, frame = v_cap.read()
            if success is False:
                continue
            if (fn in frame_list):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frames.append(frame)
        v_cap.release()
        return frames
    else:
        frames_tmp=[]
        while True:
            success, frame = v_cap.read()             
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames_tmp.append(frame)
        v_cap.release()
        for fn in frame_list:
            frames.append(frames_tmp[fn])
        return frames
    

def transform_frames(frames, model_type="rnn"):
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 100,100
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        #mean = [0.43216, 0.394666, 0.37645]
        #std = [0.22803, 0.22145, 0.216989]
#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 

    frames_tr = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_tr = test_transformer(frame)
        frames_tr.append(frame_tr)
    imgs_tensor = torch.stack(frames_tr)  
    imgs_tensor=imgs_tensor.permute(1,0,2,3)
        
    return imgs_tensor

def transform_aug_frames(frames, model_type="rnn"):
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112,112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        #mean = [0.43216, 0.394666, 0.37645]
        #std = [0.22803, 0.22145, 0.216989]
#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    test_transformer = transforms.Compose([
                transforms.RandomRotation(15), # 随机旋转-15°~15°
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 

    frames_tr = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_tr = test_transformer(frame)
        frames_tr.append(frame_tr)
    imgs_tensor = torch.stack(frames_tr)  
    imgs_tensor=imgs_tensor.permute(1,0,2,3)
        
    return imgs_tensor





def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)

#%%    dataset
class Dataset(Dataset):

    def __init__(self, train_path, frames_count,train_tag=None,transform=None):
        
        #self.root_dir = root_dir
        self.train_path = train_path
        self.frames_count=frames_count
        self.train_tag = train_tag
        self.transform = transform        
        self.size=len(train_path)
        

    def __getitem__(self, idx):
        if self.train_tag is None:
            video_path = self.train_path[idx]
            frames = get_frames(video_path,self.frames_count)
            imgs_tensor=transform_frames(frames,model_type="cnn")
            return imgs_tensor
        
        else:
            video_path = self.train_path[idx]
            if not os.path.isfile(video_path):
                print(video_path + 'does not exist!')
                return None


            frames = get_frames(video_path,self.frames_count)
            imgs_tensor=transform_frames(frames,model_type="cnn")
            label =self.train_tag[idx]
            
            return imgs_tensor,label
    
    
    def __len__(self):
        return self.size

class Dataset_aug(Dataset):

    def __init__(self, train_path, frames_count,train_tag=None,transform=None):
        
        #self.root_dir = root_dir
        self.train_path = train_path
        self.frames_count=frames_count
        self.train_tag = train_tag
        self.transform = transform        
        self.size=len(train_path)
        

    def __getitem__(self, idx):
        if self.train_tag is None:
            video_path = self.train_path[idx]
            frames = get_frames(video_path,self.frames_count)
            imgs_tensor=transform_aug_frames(frames,model_type="cnn")
            return imgs_tensor
        
        else:
            video_path = self.train_path[idx]
            if not os.path.isfile(video_path):
                print(video_path + 'does not exist!')
                return None


            frames = get_frames(video_path,self.frames_count)
            imgs_tensor=transform_aug_frames(frames,model_type="cnn")
            label =self.train_tag[idx]
            
            return imgs_tensor,label
    
    
    def __len__(self):
        return self.size

#%%
def test_submit(model,test_dataloader): 
    model.eval()
    model.to(device)
    pred_label=[]
    for data in tqdm(test_dataloader):
        if train_on_gpu:
            data = data.to(device)
        output = model(data)      
        #calculate accuracy
        _,pred=torch.max(output,1)
        pred=pred.item()
        pred_label.append(pred)
    return pred_label



def train(model,n_epochs,train_loader,valid_loader,test_loader,optimizer,criterion,scheduler,submit_name):
    train_total_acc=[]
    train_totale_losses=[]
    valid_totale_losses = []
    valid_total_acc = []
    for epoch in range(1, n_epochs+1):
        train_loss,valid_loss = 0.0,0.0
        correct_train,correct_valid = 0,0
        total_train,val_total = 0,0
        running_loss = 0.0
        train_losses,valid_losses=[],[]
        print('running epoch: {}'.format(epoch))
        print('=======training======')
        model.to(device)
        model.train()
        for data, target in tqdm(train_loader):
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _,pred=torch.max(output,1)
            correct_train += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total_train += data.size(0)
            #loss optimizer
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item()*data.size(0))  #這一個epoh的 train loss
            
        train_acc = correct_train/total_train
        train_total_acc.append(train_acc) #return 這個
        train_loss = np.average(train_losses)
        train_totale_losses.append(train_loss)
        scheduler.step()
        print(epoch, scheduler.get_last_lr()[0])
        print('\tTraining Loss: {:.6f} '.format(train_loss))
        print('\tTraining Accuracy: {:.6f}'.format(train_acc))    
        ##save the model
        model_filename = '\model'+str(epoch)+'.pth'
        torch.save(model, r'C:\Users\EE803\Desktop\HW1\model'+model_filename)
        
        #valid
        print('=======validing======')
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(valid_loader):
                if train_on_gpu:
                    data, target = data.to(device), target.to(device)
                output = model(data)       
                valid_loss = criterion(output, target)
                _,pred_val=torch.max(output,1)
                correct_valid += np.sum(np.squeeze(pred_val.eq(target.data.view_as(pred_val))).cpu().numpy())
                val_total += data.size(0)
                valid_losses.append(valid_loss.item()*data.size(0))
                
            valid_acc = correct_valid/val_total
            valid_total_acc.append(valid_acc)#return 這個
            valid_loss = np.average(valid_losses)
            valid_totale_losses.append(valid_loss)
            
            print('\tValid Loss: {:.6f} '.format(valid_loss))
            print('\tValid Accuracy: {:.6f}'.format(valid_acc)) 
        ##  輸出test 檔案 
        csv_filename = '\submit'+str(epoch)+'.csv'
        pred_label=test_submit(model,test_loader)
        submit = pd.DataFrame(data=submit_name, columns=['name'],index=None)
        submit.insert(1,column="label",value=pred_label)
        submit.to_csv(r'C:\Users\EE803\Desktop\HW1\submit' + csv_filename,index=False)
        
    return train_total_acc,train_totale_losses,valid_total_acc,valid_totale_losses,model


#%%   準備train資料與訓練


frames=8
batch_size=32


data=Dataset(train_list, frames,train_tag_list,transform=None)
#traindata=Dataset(train_true, frames,train_tag_true,transform=None)
#validdata=Dataset(val_true, frames,val_tag_true,transform=None)
train_size = int(len(data) * 14/15)
valid_size = len(data) - train_size
traindata, validdata = torch.utils.data.random_split(data, [train_size, valid_size])



train_loader = DataLoader(traindata, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(validdata, batch_size=batch_size,shuffle=True)



frames=8
test_batch_size=1
testdata=Dataset(test_list, frames,transform=None)
test_loader = DataLoader(testdata, batch_size=test_batch_size,shuffle=False)

submit_test_name = []
for i in test_list:
    submit_test_name.append(i[-9:])




model_depth=121
n_classes=39
n_input_channels=3
conv1_t_size=3
conv1_t_stride=1
no_max_pool=True


#model = generate1_model(model_depth,n_classes,n_input_channels,conv1_t_size,conv1_t_stride,no_max_pool)



model=ResNet18()
#%%

model = torch.load('C:/Users/EE803/Desktop/HW1/model/model20.pth')
epochs = 50
Learingrate=0.0002
#optimizer = torch.optim.Adam(model.parameters(), lr=Learingrate)
optimizer = SGD(model.parameters(), lr=Learingrate, momentum=0.9, weight_decay=5e-4)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,15], gamma=0.2) #learning rate decay
criterion = torch.nn.CrossEntropyLoss()

train_acc_his,train_losses_his,valid_acc_his,valid_his_losses,model1=train(model,epochs,train_loader,valid_loader,test_loader,optimizer,criterion,train_scheduler,submit_test_name)  
#torch.save(model1, r"C:\Users\user\Desktop\HW1\model.pth")

#  畫圖

x = np.linspace(0, epochs-1, epochs)
plt.xlabel('epoh', fontsize = 16)                        # 設定坐標軸標籤
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
# 繪圖並設定線條顏色、寬度、圖例
line1, = plt.plot(x, valid_acc_his, color = 'red', linewidth = 3, label = 'valid accurancy')             
line2, = plt.plot(x, train_acc_his, color = 'blue', linewidth = 3, label = 'train accurancy')
plt.legend(handles = [line1, line2], loc='upper right')
plt.show()

line3, = plt.plot(x, valid_his_losses, color = 'red', linewidth = 3, label = 'valid loss')             
line4, = plt.plot(x, train_losses_his, color = 'blue', linewidth = 3, label = 'train loss')
plt.legend(handles = [line3, line4], loc='upper right')
plt.show()





