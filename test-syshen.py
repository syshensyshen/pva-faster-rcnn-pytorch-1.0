from __future__ import absolute_import
import torch
from torchsummary import summary
import cv2
import numpy as np
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conv = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(device)

tensor = torch.empty(1, 1, 16, 16).to(device)

print(conv(tensor).shape)