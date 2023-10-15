import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from Scripts.Layers.ResidualBlock import ResidualBlock
from Scripts.Layers.AttentionModule import AttentionModule
from Scripts.Layers.Cifar10CNN_Block import Cifar10CNN_Block

class CNN_with_Attention(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        
        self.final_out = output_features
        ############ First block
        input_features=3
        output_features=6
        # CNN Block 1
        self.CNN_Block1 = Cifar10CNN_Block(
            input_features=input_features,
            output_features=output_features, 
            p=0.5, 
            pool='conv')
        # Attention Block 1
        self.residual_attention1 = AttentionModule(
            in_channels=output_features, 
            out_channels=output_features, 
            size1=(112,112),
            size2=(56,56),
            size3=(28,28))
        ############ Second blockk
        input_features=6
        output_features=12
        # CNN Block 2
        self.CNN_Block2 = Cifar10CNN_Block(
            input_features=input_features,
            output_features=output_features,
            p=0.5,
            pool='conv')
        # Attention Block 2
        self.residual_attention2 = AttentionModule(
            in_channels=output_features, 
            out_channels=output_features, 
            size1=(56,56), 
            size2=(28,28), 
            size3=(14,14))
        ############ Third blockk
        input_features=12
        output_features=24
        # CNN Block 3
        self.CNN_Block3 = Cifar10CNN_Block(
            input_features=input_features, 
            output_features=output_features, 
            p=0.1, 
            pool='conv')
        # Attention block 3
        self.residual_attention3 = AttentionModule(
            in_channels=output_features, 
            out_channels=output_features, 
            size1=(28,28), 
            size2=(14,14), 
            size3=(7,7))
        ############ Fifth blockk
        input_features=24
        output_features=48
        # Conv5_1
        self.conv5_1 = nn.Conv2d(
            in_channels=input_features,
            out_channels=output_features,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn5_1 = nn.BatchNorm2d(output_features)
        self.activation5_1 = nn.LeakyReLU()
        self.dropout5_1 = nn.Dropout2d(p=0.5)
        # Conv5_2
        input_features=output_features
        self.conv5_2 = nn.Conv2d(
            in_channels=input_features,
            out_channels=output_features,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn5_2 = nn.BatchNorm2d(output_features)
        self.activation5_2 = nn.LeakyReLU()
        self.dropout5_2 = nn.Dropout2d(p=0.5)
        
        # Last Conv block
        input_features=output_features
        output_features = self.final_out
        self.conv6 = nn.Conv2d(
            in_channels=input_features,
            out_channels=output_features,
            kernel_size=3,
            stride=1,
            padding=1)
        self.maxpool6 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=4840,
                               out_features=10)
        self.activation6_1 = nn.Sigmoid()
        
    def forward(self, x):
        
        # First Block
        out = self.CNN_Block1(x)
        out = self.residual_attention1(out)
        # Second Block
        out = self.CNN_Block2(out)
        out = self.residual_attention2(out)
        # Third Block
        out = self.CNN_Block3(out)
        out = self.residual_attention3(out)
        # Fifth Block
        out = self.conv5_1(out)
        out = self.bn5_1(out)
        out = self.activation5_1(out)
        out = self.dropout5_1(out)
        out = self.conv5_2(out)
        out = self.bn5_2(out)
        out = self.activation5_2(out)
        out = self.dropout5_2(out)
        # Last Block
        out = self.conv6(out)
        out = self.maxpool6(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.activation6_1(out)

        return out