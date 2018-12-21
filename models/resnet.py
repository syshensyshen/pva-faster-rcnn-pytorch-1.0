import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torchsummary import summary


class ConvBatchNorm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, group, bias,
                 output_relu=True):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class BlockDownSampleConv(nn.Module):
    '''
    block shape:
          0
        /   \
    1 conv   \
       /      \
  2 conv    conv
      /         \
 3  conv         \
      \          /
       \        /
        \      /
        eltwise
    in_planes1--1==in_planes2--2;;in_planes3--3
    '''
    '''
    block shape:
          0
        /   \
    1 conv   \
       /      \
  2 conv       \
      /         \
 3  conv         \
      \          /
       \        /
        \      /
        eltwise
    in_planes1--1;;in_planes2--2==in_planes3--3
    '''
    def __init__(self, in_planes, out_planes, bias, stride, downsample):
                 super(BlockDownSampleConv, self).__init__()
                 self.leftconv1 = ConvBatchNorm(in_planes, out_planes/2, kernel_size=1,
                                                stride=1, padding=0, group=1, bias=bias)
                 self.leftconv2 = ConvBatchNorm(out_planes/2, out_planes/2, kernel_size=3,
                                                stride=stride, padding=1, group=32,
                                                bias=bias)
                 self.leftconv3 = ConvBatchNorm(out_planes/2, out_planes, kernel_size=1,
                                                stride=1, padding=0, group=1, bias=bias)
                 if downsample:
                     self.rightconv = ConvBatchNorm(in_planes, out_planes, kernel_size=1,
                                                    stride=stride, padding=0, group=1, bias=bias)
                 self.downsample = downsample

    def forward(self, x):
        y1 = self.leftconv1(x)
        y1 = self.leftconv2(y1)
        y1 = self.leftconv3(y1)
        if self.downsample:
            y2 = self.rightconv(x)                  
            return y1 + y2
        else:
            return y1 + x

class ResNet50Body(nn.Module):
    def __init__(self):
        super(ResNet50Body, self).__init__()
        self.conv1 = ConvBatchNorm(3, 64, kernel_size=7, stride=2, padding=3, group=1, bias=False)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = BlockDownSampleConv(64, 256, False, 1, True)
        self.block2 = BlockDownSampleConv(256, 256, False, 1, False)
        self.block3 = BlockDownSampleConv(256, 256, False, 1, False)
        self.block4 = BlockDownSampleConv(256, 512, False, 2, True)
        self.block5 = BlockDownSampleConv(512, 512, False, 1, False)
        self.block6 = BlockDownSampleConv(512, 512, False, 1, False)
        self.block7 = BlockDownSampleConv(512, 512, False, 1, False)
        self.block8 = BlockDownSampleConv(512, 1024, False, 2, True)
        self.block9 = BlockDownSampleConv(1024, 1024, False, 1, False)
        self.block10 = BlockDownSampleConv(1024, 1024, False, 1, False)
        self.block11 = BlockDownSampleConv(1024, 1024, False, 1, False)
        self.block12 = BlockDownSampleConv(1024, 1024, False, 1, False)
        self.block13 = BlockDownSampleConv(1024, 1024, False, 1, False)
        self.block14 = BlockDownSampleConv(1024, 2048, False, 2, True)
        self.block15 = BlockDownSampleConv(2048, 2048, False, 1, False)
        self.block16 = BlockDownSampleConv(2048, 2048, False, 1, False)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model=ResNet50Body().to(device)
summary(model, (3, 224, 224))
