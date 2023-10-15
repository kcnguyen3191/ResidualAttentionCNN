############## The code adapted from
####### https://github.com/liudaizong/Residual-Attention-Network/blob/master/model/basic_layers.py
############## Residual Block is described in the paper by: He et. al. 2015, Deep Residual Learning for Image Recognition
####### https://arxiv.org/pdf/1512.03385.pdf

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, 
                               output_channels, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(input_channels, 
                               output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, 
                               output_channels, 1, 1, bias = False)
        
    def forward(self, x):
        
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out