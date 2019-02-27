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

        P2_x = self.P3_1(C2)
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