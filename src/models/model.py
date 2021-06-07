import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, BatchNorm1d
import torch.nn.functional as F


def compute_conv_dim(dim_size, kernel_size, padding_conv1=0, stride_conv=1,n_convs=1):
    for ii in range(n_convs):
        if ii == 0:
            new_dim = int((dim_size - kernel_size + 2 * padding_conv1) / stride_conv + 1)
        else:
            new_dim = int((new_dim - kernel_size + 2 * padding_conv1) / stride_conv + 1)
    return new_dim


class MyAwesomeModel(nn.Module):
    def __init__(self,image_dim = 28,kernel_size = 5,filters = 32,fc_features = 128,n_classes = 10):
        super().__init__()

        self.image_dim = image_dim
        self.kernel_size = kernel_size
        self.filters = filters
        self.fc_features = fc_features
        self.n_classes = n_classes
        
        self.conv1 = Conv2d(in_channels = 1,
                             out_channels = self.filters,
                             kernel_size = self.kernel_size)
        
        self.conv2 = Conv2d(in_channels = self.filters,
                             out_channels = 2*self.filters,
                             kernel_size = self.kernel_size)

        self.conv3 = Conv2d(in_channels = 2*self.filters,
                             out_channels = 2*self.filters,
                             kernel_size=self.kernel_size)
        
        self.conv_dim = compute_conv_dim(self.image_dim,self.kernel_size,n_convs=3)

        self.fc1 = nn.Linear(self.conv_dim**2*2*self.filters, self.fc_features)

        self.fc2 = nn.Linear(self.fc_features, self.n_classes)
        
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)

        return x
    