
import torch
import torch.nn as nn
from models.pvanet import PVANetFeat
from models.lite import PVALiteFeat
import torch.nn.functional as F

class pvaHyper(PVANetFeat):
    '''
    '''
    def __init__(self):
        super(pvaHyper, self).__init__()

    def forward(self, input):
        x0 = self.conv1(input)
        x1 = self.conv2(x0)  # 1/4 feature
        x2 = self.conv3(x1)  # 1/8
        x3 = self.conv4(x2)  # 1/16
        x4 = self.conv5(x3)  # 1/32
        downsample = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=0).view(x2.size(0), -1)
        upsample = F.interpolate(x4, scale_factor=2, mode="nearest")
        features = torch.cat((downsample, x3, upsample), 1)
        return features

class liteHyper(PVALiteFeat):
    def __init__(self):
        super(liteHyper, self).__init__()
    def forward(self, input):
        x1 = self.conv1(input) # 1/2 feature
        x2 = self.conv2(x1) # 1/4 feature
        x2 = self.conv3(x2) # 1/8 feature
        x3 = self.Inception3a(x2) # 1/16 feature
        x3 = self.Inception3b(x3)
        x3 = self.Inception3c(x3)
        x3 = self.Inception3d(x3)
        x3 = self.Inception3e(x3)
        x4 = self.Inception4a(x3) # 1/32 feature
        x4 = self.Inception4b(x4)
        x4 = self.Inception4c(x4)
        x4 = self.Inception4d(x4)
        x4 = self.Inception4e(x4)
        downsample = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)
        upsample = F.interpolate(x4, scale_factor=2, mode="nearest")
        features = torch.cat((downsample, x3, upsample), 1)
        return features