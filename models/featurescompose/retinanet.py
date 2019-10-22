from __future__ import absolute_import
import torch.nn as nn
import torch
import math
import time
import torch.nn.functional as F
import numpy as np
import sys

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class DilatedPyramidFeaturesEx(nn.Module):
    def __init__(self, cfg):
        C3_size, C4_size, C5_size = cfg.in_channels
        feature_size = cfg.out_channels

        super(DilatedPyramidFeaturesEx, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        #### features using diff mothod in deep
        self.P3_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P5_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P6_down1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.P3_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P4_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P5_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P6_down2 = nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1)

        ###### dilate convolution for deep
        self.Dilate3_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate4_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate5_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=2)
        self.Dilate6_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=3)
        self.Dilate7_x = nn.Conv2d(feature_size, feature_size, 3, padding=1, dilation=4)

        init_weights(self.modules())

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        P3_d1 = self.P3_down1(P3_x)
        P3_d2 = self.P3_down2(P3_x)
        # print('P4_x', P4_x.shape, P3_d.shape)
        P4_x = P4_x + P3_d1 + P3_d2
        P4_d1 = self.P4_down1(P4_x)
        P4_d2 = self.P4_down2(P4_x)
        # print('P5_x', P5_x.shape, P4_d.shape)
        P5_x = P5_x + P4_d1 + P4_d2
        P5_d1 = self.P5_down1(P5_x)
        P5_d2 = self.P5_down2(P5_x)
        # print('P6_x', P6_x.shape, P5_d.shape)
        P6_x = P6_x + P5_d1 + P5_d2
        P6_d1 = self.P6_down1(P6_x)
        P6_d2 = self.P6_down2(P6_x)
        # print('P7_x', P7_x.shape, P6_d.shape)
        P7_x = P7_x + P6_d1 + P6_d2

        ### dilate convolution
        P3_x = self.Dilate3_x(P3_x)
        P4_x = self.Dilate3_x(P4_x)
        P5_x = self.Dilate3_x(P5_x)
        P6_x = self.Dilate3_x(P6_x)
        P7_x = self.Dilate3_x(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class PyramidFeaturesEx(nn.Module):
    def __init__(self, cfg):
        C3_size, C4_size, C5_size = cfg.in_channels
        feature_size = cfg.out_channels
        super(PyramidFeaturesEx, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P3_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P5_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P6_down = nn.MaxPool2d(3, stride=2, padding=1)

        init_weights(self.modules())

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        P3_d = self.P3_down(P3_x)
        # print('P4_x', P4_x.shape, P3_d.shape)
        P4_x = P4_x + P3_d
        P4_d = self.P4_down(P4_x)
        # print('P5_x', P5_x.shape, P4_d.shape)
        P5_x = P5_x + P4_d
        P5_d = self.P5_down(P5_x)
        # print('P6_x', P6_x.shape, P5_d.shape)
        P6_x = P6_x + P5_d
        P6_d = self.P6_down(P6_x)
        # print('P7_x', P7_x.shape, P6_d.shape)
        P7_x = P7_x + P6_d

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class PyramidFeatures(nn.Module):
    def __init__(self, cfg):
        _, C3_size, C4_size, C5_size = cfg.model.network.hyper_inchannels
        feature_size = cfg.model.network.rpn_cin
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled   = F.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        init_weights(self.modules())

    def forward(self, inputs):
        _, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2, mode="nearest")
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2, mode="nearest")
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]



def get_model(config):
    return PyramidFeatures(config)