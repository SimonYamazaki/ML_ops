import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BatchNorm1d, Conv2d, Dropout, Linear, MaxPool2d


def compute_conv_dim(dim_size, kernel_size, padding_conv1=0, stride_conv=1, n_convs=1):
    for ii in range(n_convs):
        if ii == 0:
            new_dim = int(
                (dim_size - kernel_size + 2 * padding_conv1) / stride_conv + 1
            )
        else:
            new_dim = int((new_dim - kernel_size + 2 * padding_conv1) / stride_conv + 1)
    return new_dim


class MyAwesomeModel(nn.Module):
    def __init__(
        self,
        image_dim=28,
        kernel_size=5,
        filters=32,
        fc_features=128,
        n_classes=10,
        get_feature_layer=False,
    ):
        super().__init__()

        self.image_dim = image_dim
        self.kernel_size = kernel_size
        self.filters = filters
        self.fc_features = fc_features
        self.n_classes = n_classes
        self.get_feature_layer = get_feature_layer

        self.conv1 = Conv2d(
            in_channels=1, out_channels=self.filters, kernel_size=self.kernel_size
        )

        self.conv2 = Conv2d(
            in_channels=self.filters,
            out_channels=2 * self.filters,
            kernel_size=self.kernel_size,
        )

        self.conv3 = Conv2d(
            in_channels=2 * self.filters,
            out_channels=2 * self.filters,
            kernel_size=self.kernel_size,
        )

        self.conv_dim = compute_conv_dim(self.image_dim, self.kernel_size, n_convs=3)

        self.fc1 = nn.Linear(self.conv_dim ** 2 * 2 * self.filters, self.fc_features)

        self.fc2 = nn.Linear(self.fc_features, self.n_classes)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.logSM = nn.LogSoftmax(dim=1)

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        x1 = self.fc1(x)

        x = self.relu(x1)
        x = self.dropout3(x)

        x1 = self.fc2(x)
        x = self.logSM(x1)

        if self.get_feature_layer:
            return x1
        else:
            return x


"""
image_dim = 28
kernel_size = 5
filters = 32
fc_features = 128
n_classes = 10 

from src.data.data import mnist

model = MyAwesomeModel(image_dim,kernel_size,filters,fc_features,n_classes)
trainloader, _ = mnist()
n_batches = len(trainloader)
model.train()

criterion = nn.NLLLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for images, labels in trainloader:
    print(images.shape)
    log_ps = model(images)
"""
