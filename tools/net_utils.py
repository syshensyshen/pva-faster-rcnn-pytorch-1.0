
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy
import colorsys
import random
import torchvision.models as models
#from models.config import cfg
import cv2
import pdb

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def get_current_lr(optimizer):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = lr if param_group['lr'] < lr else param_group['lr']
    return lr