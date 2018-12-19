# -*- coding: utf-8 -*-
# @Author  : syshen 
# @date: 2018/12/19 11:14
# @File    : modeified Denset using two liner layer


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from random import shuffle
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

class Densenet_(nn.Module):
    def __init__(self, cls_num):
        super(Densenet_, self).__init__()
        Backstone = models.densenet201(pretrained=True)
        num_ftrs = Backstone.features
        self.BackstoneDim = 1920
        self.featureDim = 1280
        add_block = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.BackstoneDim, self.featureDim )),
            ('fc_relu', nn.LeakyReLU(inplace=True)),
            ('classifier', nn.Linear(self.featureDim, cls_num, bias=True))]))
        self.Backstone = Backstone.features
        self.add_block = add_block
        init.xavier_uniform(self.add_block.fc1.weight)
        init.constant(self.add_block.fc1.bias, 1.0)
        init.normal(self.add_block.classifier.weight, mean=0, std=0.001)
        init.constant(self.add_block.fc1.bias, 1.0)
        #init.xavier_uniform(self.add_block.classifier.weight)
        #init.constant(self.add_block.fc1.bias, 1.0)
        #init.normal(self.add_block.fc1.weight, mean=0, std=0.01)
        #init.constant(self.add_block.fc1.bias, 0.0)
        #init.normal(self.add_block.classifier.weight, mean=0, std=0.001)
        #init.constant(self.add_block.fc1.bias, 0.0)
        #init.kaiming_uniform_(self.add_block.fc1.weight)
        #init.constant(self.add_block.fc1.bias, 0.2)
        #init.kaiming_uniform_(self.add_block.classifier.weight)
        #init.constant(self.add_block.classifier.bias, 1.0)

    def forward(self, input):
        x = self.Backstone(input)
        H = x.shape[2]
        W = x.shape[3]
        x_relu = F.relu(x, inplace=True)
        x_relu = F.avg_pool2d(x_relu, (H, W), stride=1).view(x.size(0), -1)
        x = self.add_block(x_relu)
        return x
