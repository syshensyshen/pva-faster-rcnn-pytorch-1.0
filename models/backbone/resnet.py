from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import torch.nn as nn

from .basic_modules import BasicBlock, Bottleneck
from tools.checkpoint import load_checkpoint


class ResNet(nn.Module):


    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }


    model_urls = {
        18: 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
        34: 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
        50: 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        101: 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
        152: 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
    }


    def __init__(self, cfg):
        super(ResNet, self).__init__()
        # assert self.depth in self.arch_settings, 'resnet depth is not exists'
        self.depth=cfg.depth
        block, layers = self.arch_settings[self.depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.__init_weights__(cfg.pretrained)


    def __init_weights__(self, pretrained=True):
        if pretrained:
            load_checkpoint(self, self.model_urls[self.depth])
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if not m.bias is None:
                        m.bias.data.zero_()
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, images):
        x0 = self.conv1(images) # 1/2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0) # 1/4

        x1 = self.layer1(x1) # 1/4
        x2 = self.layer2(x1) # 1/8
        x3 = self.layer3(x2) # 1/16
        x4 = self.layer4(x3) # 1/32

        all_features = [x1, x2, x3, x4]

        return all_features


def get_model(config):
    model = ResNet(config)
    return model


if __name__ == "__main__":
    pass

    # model = ResNet()   
    # model = model.cuda()
    # inputs = torch.randn(1, 3, 512, 512).cuda().float()
    # outputs = model(inputs)
    # for i, output in enumerate(outputs):
    #     print("outputs[{}] size: {}".format(i, output.shape))