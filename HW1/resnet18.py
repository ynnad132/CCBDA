import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        #input size 16*3*112*112
        self.conv1=nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3),bias=False)
        self.bn1=nn.BatchNorm3d(64)
        self.relu=nn.ReLU(inplace=True)
        #self.pool1=nn.MaxPool3d(kernel_size=3,stride=2,padding=1,dilation=1)
#input size 16*64*56*56
        #block2_1
        self.conv2_1=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn2_1=nn.BatchNorm3d(64)
        self.conv2_2=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn2_2=nn.BatchNorm3d(64)
#input size 16*64*56*56
        #block2_2
        self.conv2_3=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn2_3=nn.BatchNorm3d(64)
        self.conv2_4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn2_4=nn.BatchNorm3d(64)
#input size 16*64*56*56
        #block3_1
        self.conv3_1=nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bn3_1=nn.BatchNorm3d(128)
        self.conv3_2=nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn3_2=nn.BatchNorm3d(128)
        self.downsample1 = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,1,1), stride=(2,2,2), bias=False),
                nn.BatchNorm3d(128)
                )
#input size 8*128*28*28
        #block3_2
        self.conv3_3=nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn3_3=nn.BatchNorm3d(128)
        self.conv3_4=nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn3_4=nn.BatchNorm3d(128)
#input size 8*128*28*28
        #block4_1
        self.conv4_1=nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bn4_1=nn.BatchNorm3d(256)
        self.conv4_2=nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn4_2=nn.BatchNorm3d(256)
        self.downsample2 = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1,1,1), stride=(2,2,2), bias=False),
                nn.BatchNorm3d(256)
                )
    #input size 4*256*14*14
        #block4_2
        self.conv4_3=nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn4_3=nn.BatchNorm3d(256)
        self.conv4_4=nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn4_4=nn.BatchNorm3d(256)
#input size 4*256*14*14
        #block5_1
        self.conv5_1=nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bn5_1=nn.BatchNorm3d(512)
        self.conv5_2=nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn5_2=nn.BatchNorm3d(512)
        self.downsample3 = nn.Sequential(
                nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1,1,1), stride=(2,2,2), bias=False),
                nn.BatchNorm3d(512)
                )
    #input size 2*512*7*7
    #block5_2
        self.conv5_3=nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn5_3=nn.BatchNorm3d(512)
        self.conv5_4=nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn5_4=nn.BatchNorm3d(512)

    #self.avgpool=nn.AdaptiveAvgPool3d(kernel_size=7,stride=1,padding=0)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1=nn.Linear(128,39)
        #self.lstm=nn.LSTM(input_size=3,hidden_size=6,num_layers=1,bias=False,)
        #self.dp1=nn.Dropout(0.2)
        #self.fc2 = nn.Linear(128,39)
        #self.out = nn.Softmax(dim=1)
    
    def forward(self, input):
        output=self.conv1(input)
        output=F.relu(self.bn1(output))
        #output=self.pool1(output)
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
        #residul4_1=self.downsample2(output)
        #output=self.conv4_1(output)
        #output=F.relu(self.bn4_1(output))
        #output=self.conv4_2(output)
        #output=self.bn4_2(output)
        #output+=residul4_1
        #output=F.relu(output)
        #block4_2
        #residul4_2=output
        #output=self.conv4_3(output)
        #output=F.relu(self.bn4_3(output))
        #output=self.conv4_4(output)
        #output=self.bn4_4(output)
        #output+=residul4_2
        #output=F.relu(output)
    
        #block5_1
        #residul5_1=self.downsample3(output)
        #output=self.conv5_1(output)
        #output=F.relu(self.bn5_1(output))
        #output=self.conv5_2(output)
        #output=self.bn5_2(output)
        #output+=residul5_1
        #output=F.relu(output)
        #block3_2
        #residul5_2=output
        #output=self.conv5_3(output)
        #output=F.relu(self.bn5_3(output))
        #output=self.conv5_4(output)
        #output=self.bn5_4(output)
        #output+=residul5_2
        #output=F.relu(output)
    
        output=self.avgpool(output)
        output = output.view(output.size(0), -1)
        output=self.fc1(output)
        #output = self.dp1(output)
        #output = self.fc2(output)
        #output=self.out(output)
        return output