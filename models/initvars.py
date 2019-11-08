# -*- coding: utf-8 -*-
'''
# @Author  : syshen 
# @date    : 2019/1/15
'''
import torch
import torch.nn as nn
from models.pvanet import PVANetFeat
from models.lite import PVALiteFeat
from models.mobilenet_v2 import MobileNet2
import torch.nn.functional as F
import math

def initvars(modules):
    # Copied from vision/torchvision/models/resnet.py
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



