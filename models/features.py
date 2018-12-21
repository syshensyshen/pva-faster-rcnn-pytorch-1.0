
import torch
import torch.nn as nn
from models.pvanet import PVANetFeat
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