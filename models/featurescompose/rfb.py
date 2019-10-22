from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import math

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.001)
                m.bias.data.fill_(0)

    def forward(self, feats):
        b0 = self.branch0(feats)
        b1 = self.branch1(feats)
        b2 = self.branch2(feats)
        b3 = self.branch3(feats)
        outputs = self.conv_cat(torch.cat([b0, b1, b2, b3], dim=1))
        outputs += self.conv_res(feats)

        return outputs


def get_model(config):
    return RFB(config)