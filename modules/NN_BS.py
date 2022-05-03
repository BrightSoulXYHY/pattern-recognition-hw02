import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_FCx1(nn.Module):
    def __init__(self, input_size=32*32*3, hidden_size=5000, num_classes=10):
        super(NN_FCx1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class NN_CNN(nn.Module):
    def __init__(self):
        super(NN_CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296,128)
        self.fc2 = nn.Linear(128,10)      

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1,36*6*6)
        x=F.relu(self.fc2(F.relu(self.fc1(x))))
        return x



class NN_CNN_AvgPool(nn.Module):
    def __init__(self):
        super(NN_CNN_AvgPool, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.aap=nn.AdaptiveAvgPool2d(1)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = x.view(-1, 16 * 5 * 5)
        x = self.aap(x)
        #print(x.shape)
        #x = F.relu(self.fc2(x))
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc3(x)
        return x
    


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.cnn=nn.Sequential(

            #Conv1
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=2,padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #Conv2
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),
            #Conv3
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            #Conv4
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            #Conv5
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.lin=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,10),
            nn.LogSoftmax()
        )
    def forward(self,x):
        x=self.cnn(x)
        x=self.lin(x)
        return x

class NN_Stacking(nn.Module):
    def __init__(self, input_size=40, hidden_size=120, num_classes=10):
        super(NN_Stacking, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
