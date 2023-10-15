import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from Scripts.Layers.ResidualBlock import ResidualBlock

class Cifar10CNN_Block(nn.Module):
    def __init__(self, input_features, output_features, p=0.5, pool='max'):
        super().__init__()
        
        self.p = p
        self.input_features=input_features
        self.output_features=output_features
        self.pool_type = pool
        
        if self.pool_type == 'max':
            self.pool_fun = nn.MaxPool2d
        if self.pool_type == 'avg':
            self.pool_fun = nn.AvgPool2d
        
        # First conv block
        self.conv1 = nn.Conv2d(in_channels=self.input_features,
                              out_channels=self.output_features,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1 = nn.BatchNorm2d(self.output_features)
        self.activation1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout2d(p=self.p)
        # Second conv block
        self.input_features = self.output_features
        self.conv2 = nn.Conv2d(in_channels=self.input_features,
                              out_channels=self.output_features,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn2 = nn.BatchNorm2d(self.output_features)
        self.activation2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout2d(p=self.p)
        # Pooling
        if self.pool_type == 'max' or self.pool_type == 'avg':
            self.pool = self.pool_fun(kernel_size=2,
                                     stride=2)
        else:
            self.pool = nn.Conv2d(in_channels=self.input_features,
                                  out_channels=self.output_features,
                                  kernel_size=2,
                                  stride=2,
                                  padding=0)
            self.bn3 = nn.BatchNorm2d(self.output_features)
            self.activation3 = nn.LeakyReLU()
            
    def forward(self, x):
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        # Pooling block
        if self.pool_type == 'max' or self.pool_type == 'avg':
            out = self.pool(out)
        else:
            out = self.pool(out)
            out = self.bn3(out)
            out = self.activation3(out)
        
        return out