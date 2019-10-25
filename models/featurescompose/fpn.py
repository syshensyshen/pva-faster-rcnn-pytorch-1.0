
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import math

from ..backbone.basic_modules import Conv2d, initvars


class DilatedPyramidFeaturesEx(nn.Module):

    def __init__(self, inchannels, output_channels):
        C2_size, C3_size, C4_size, C5_size = inchannels
        feature_size = output_channels

        super(DilatedPyramidFeaturesEx, self).__init__()

        self.P2_1 = Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.P6_1 = Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # self.P6_2 = Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)

        #### features using diff mothod in deep
        self.P2_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P3_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down1 = nn.MaxPool2d(3, stride=2, padding=1)
        #self.P5_down1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.P2_down2 = Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P3_down2 = Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        self.P4_down2 = Conv2d(feature_size, feature_size, 3, stride=2, padding=1)
        #self.P5_down2 = Conv2d(feature_size, feature_size, 3, stride=2, padding=1)

        ###### dilate convolution for deep
        self.Dilate2_x = Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate3_x = Conv2d(feature_size, feature_size, 3, padding=1, dilation=1)
        self.Dilate4_x = Conv2d(feature_size, feature_size, 3, padding=1, dilation=2)
        self.Dilate5_x = Conv2d(feature_size, feature_size, 3, padding=1, dilation=3)
        # self.Dilate6_x = Conv2d(feature_size, feature_size, 3, padding=1, dilation=4)

        initvars(self.modules())

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        # P6_x = self.P6_1(C5)
        # P6_x = self.P6_2(P6_x)

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

        P2_d1 = self.P2_down1(P2_x)
        P2_d2 = self.P2_down2(P2_x)
        # print('P4_x', P4_x.shape, P3_d.shape)
        P3_x = P3_x + P2_d1 + P2_d2
        P3_d1 = self.P3_down1(P3_x)
        P3_d2 = self.P3_down2(P3_x)
        # print('P5_x', P5_x.shape, P4_d.shape)
        P4_x = P4_x + P3_d1 + P3_d2
        P4_d1 = self.P4_down1(P4_x)
        P4_d2 = self.P4_down2(P4_x)
        # print('P6_x', P6_x.shape, P5_d.shape)
        P5_x = P5_x + P4_d1 + P4_d2
        # P5_d1 = self.P5_down1(P5_x)
        # P5_d2 = self.P5_down2(P5_x)
        # print('P7_x', P7_x.shape, P6_d.shape)
        #P6_x = P6_x + P5_d1 + P5_d2

        ### dilate convolution
        P2_x = self.Dilate2_x(P3_x)
        P3_x = self.Dilate3_x(P3_x)
        P4_x = self.Dilate4_x(P4_x)
        P5_x = self.Dilate5_x(P5_x)
        # P6_x = self.Dilate6_x(P6_x)

        #return [P2_x, P3_x, P4_x, P5_x, P6_x]
        return [P2_x, P3_x, P4_x, P5_x]


class PyramidFeaturesEx(nn.Module):
    def __init__(self, inchannels, output_channels):
        C2_size, C3_size, C4_size, C5_size = inchannels
        feature_size = output_channels
        super(PyramidFeaturesEx, self).__init__()

        self.P2_1 = Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # self.P6_1 = Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # self.P6_2 = Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)

        self.P2_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P3_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P4_down = nn.MaxPool2d(3, stride=2, padding=1)
        self.P5_down = nn.MaxPool2d(3, stride=2, padding=1)

        initvars(self.modules())

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        # P6_x = self.P6_1(C5)
        # P6_x = self.P6_2(P6_x)

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
        # P5_d = self.P5_down(P5_x)
        # P6_x = P6_x + P5_d

        #return [P2_x, P3_x, P4_x, P5_x, P6_x]  # 1/4, 1/8, 1/16, 1/32, 1/64
        return [P2_x, P3_x, P4_x, P5_x]  # 1/4, 1/8, 1/16, 1/32


class PyramidFeatures(nn.Module):

    def __init__(self, inchannels, output_channels):
        C2_size, C3_size, C4_size, C5_size = inchannels
        feature_size = output_channels
        super(PyramidFeatures, self).__init__()

        self.P2_1 = Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6_1 = Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6_2 = Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)

        initvars(self.modules())

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        P6_x = self.P6_1(C5)
        P6_x = self.P6_2(P6_x)

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

        return [P2_x, P3_x, P4_x, P5_x, P6_x]  # 1/4, 1/8, 1/16, 1/32, 1/64
        # return [P2_x, P3_x, P4_x, P5_x]  # 1/4, 1/8, 1/16, 1/32, 1/64



class PyramidFeatures(nn.Module):

    def __init__(self, cfg):
        C2_size, C3_size, C4_size, C5_size = cfg.inchannels
        feature_size = cfg.output_channels
        super(PyramidFeatures, self).__init__()

        self.P2_1 = Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6_1 = Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6_2 = Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)

        # init_weights(self.modules())

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        P6_x = self.P6_1(C5)
        P6_x = self.P6_2(P6_x)

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

        return [P2_x, P3_x, P4_x, P5_x, P6_x]  # 1/4, 1/8, 1/16, 1/32, 1/64
        # return [P2_x, P3_x, P4_x, P5_x]  # 1/4, 1/8, 1/16, 1/32, 1/64


def get_model(config):
    return PyramidFeatures(config)
