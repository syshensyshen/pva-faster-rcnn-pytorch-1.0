'''
# author: syshen 
# date  : 2019/02-2019/03
'''
from __future__ import absolute_import
import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import sys
from torch.autograd import Variable
import pdb

class DilatedPyramidFeaturesEx(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(DilatedPyramidFeaturesEx, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        #### features using diff mothod in deep
        self.P2_down1 = nn.MaxPool2d(3, stride=2, padding=1)        
        self.P3_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        #self.P5_down1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.P2_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)        
        self.P3_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P4_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        #self.P5_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)

        ###### dilate convolution for deep
        self.Dilate2_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate3_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate4_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=2)
        self.Dilate5_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=3)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_upsampled_x = F.interpolate(P3_x, scale_factor=2, mode="nearest")
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P2_d1 = self.P3_down1(P2_x)
        P2_d2 = self.P3_down2(P2_x)
        P3_x = P3_x + P2_d1 + P2_d2
        P3_d1 = self.P3_down1(P3_x)
        P3_d2 = self.P3_down2(P3_x)
        P4_x = P4_x + P3_d1 + P3_d2
        P4_d1 = self.P4_down1(P4_x)
        P4_d2 = self.P4_down2(P4_x)
        P5_x = P5_x + P4_d1 + P4_d2

        ### dilate convolution
        P2_x = self.Dilate2_x(P2_x)
        P3_x = self.Dilate3_x(P3_x)
        P4_x = self.Dilate3_x(P4_x)
        P5_x = self.Dilate3_x(P5_x)

        return [P2_x, P3_x, P4_x, P5_x]

class PyramidFeaturesEx(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeaturesEx, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P2_down = nn.MaxPool2d(3, stride=2, padding=1)        
        self.P3_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P5_down = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = F.interpolate(P3_x, scale_factor=2, mode="nearest")
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P2_d = self.P2_down(P2_x)       
        P3_x = P3_x + P2_d
        P3_d = self.P3_down(P3_x)
        P4_x = P4_x + P3_d
        P4_d = self.P4_down(P4_x)
        P5_x = P5_x + P4_d

        return [P2_x, P3_x, P4_x, P5_x]

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        normal_init(self.P2_1, 0, 0.01)
        normal_init(self.P2_2, 0, 0.01)
        normal_init(self.P3_1, 0, 0.01)
        normal_init(self.P3_2, 0, 0.01)
        normal_init(self.P4_1, 0, 0.01)
        normal_init(self.P4_2, 0, 0.01)
        normal_init(self.P5_1, 0, 0.01)
        normal_init(self.P5_2, 0, 0.01)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5) 
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")      
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        
        P4_x = P5_upsampled_x + P4_x     
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")   
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = F.interpolate(P3_x, scale_factor=2, mode="nearest")
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)        
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P3_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x]

class hyper_features(nn.Module):
    def __init__(self, down_channels):
        super(hyper_features,self).__init__()
        self.downsample = nn.Conv2d(down_channels, down_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, ims):
        down=self.downsample(ims[0])
        up=F.interpolate(ims[2], scale_factor=2, mode="nearest")
        
        return torch.cat((down, up, ims[1]), 1)

class resnethyper(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(resnethyper, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == 'BasicBlock':
            self.fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == 'Bottleneck':
            self.fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = DilatedPyramidFeaturesEx(self.fpn_sizes[0], self.fpn_sizes[1], self.fpn_sizes[2], self.fpn_sizes[3])
        self.downx2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.downx3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img_batch):

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)
        features = self.fpn([x1, x2, x3, x4])

        d2 = self.downx2(features[0])
        d3 = self.downx3(features[1])
        up5 = F.interpolate(features[3], scale_factor=2, mode="nearest")

        feat_ = torch.cat((d2, d3, features[2], up5), 1)

        return feat_

class resnethyperex(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(resnethyperex, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == 'BasicBlock':
            self.fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == 'Bottleneck':
            self.fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(self.fpn_sizes[0], self.fpn_sizes[1], self.fpn_sizes[2], self.fpn_sizes[3])
        self.hyper_features = hyper_features(256)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img_batch):

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)
        features = self.fpn([x1, x2, x3, x4])

        feat_1 = self.hyper_features([features[0], features[1], features[2]])
        feat_2 = self.hyper_features([features[1], features[2], features[3]])

        return [feat_1, feat_2]

