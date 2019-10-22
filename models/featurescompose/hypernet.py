from __future__ import absolute_import
import math
import time
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..backbone.basic_modules import Conv2d, initvars


class HyerNet(nn.Module):
    def __init__(self, cfg):

        C2_size, C3_size, C4_size, C5_size = cfg.inchannels
        self.hyper_dim = cfg.hyper_dim
        self.feature_size = cfg.output_channels
        super(HyerNet, self).__init__()
        
        self.C2 = Conv2d(C2_size, self.feature_size, kernel_size=1)
        self.C3 = Conv2d(C3_size, self.feature_size, kernel_size=1)
        self.C4 = Conv2d(C4_size, self.feature_size, kernel_size=1)
        self.C5 = Conv2d(C5_size, self.feature_size, kernel_size=1)

        self.conv_last = Conv2d(self.feature_size, self.feature_size, kernel_size=1)

        initvars(self.modules())

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        C2 = self.C2(C2)
        C3 = self.C3(C3)
        C4 = self.C4(C4)
        C5 = self.C5(C5)
        if self.hyper_dim == 0:
            C5 = F.interpolate(C5, scale_factor=2)
            C4 = F.interpolate(C4 + C5, scale_factor=2)
            C3 = F.interpolate(C3 + C4, scale_factor=2)
            feat = C3 + C2
        elif self.hyper_dim == 1:
            C5 = F.interpolate(C5, scale_factor=2)
            C4 = F.interpolate(C4 + C5, scale_factor=2)
            C2 = F.interpolate(C2, scale_factor=0.5)
            feat = C2 + C3 + C4
        elif self.hyper_dim == 2:
            C5 = F.interpolate(C5, scale_factor=2)
            C2 = F.interpolate(C2, scale_factor=0.5)
            C3 = F.interpolate(C2 + C3, scale_factor=0.5)
            feat = C3 + C4 + C5
        elif self.hyper_dim == 3:
            C2 = F.interpolate(C2, scale_factor=0.5)
            C3 = F.interpolate(C2 + C3, scale_factor=0.5)
            C4 = F.interpolate(C3 + C4, scale_factor=0.5)
            feat = C4 + C5
        else:
            assert True, 'error hyper mixture dim'
        

        feat = self.conv_last(feat)

        return feat


def get_model(config):
    return HyerNet(config)