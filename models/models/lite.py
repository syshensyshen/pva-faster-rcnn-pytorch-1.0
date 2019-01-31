import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.pva_faster_rcnn import pva_faster_rcnn 

def initvars(modules):
    # Copied from vision/torchvision/models/resnet.py
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class ConvBn(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv(input)

class Inception3a(nn.Module):
    def __init__(self):
        super(Inception3a, self).__init__()
        self.branch11 = ConvBn(96, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, stride=2, padding=1)
        self.branch21 = ConvBn(96, 16, kernel_size=1)
        self.branch22 = ConvBn(16, 64, kernel_size=3, stride=2, padding=1)
        self.branch31 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch32 = ConvBn(96, 96, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        right = self.branch32(right)
        return torch.cat((left, midle, right), 1)

class Inception3b(nn.Module):
    def __init__(self):
        super(Inception3b, self).__init__()
        self.branch11 = ConvBn(192, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(192, 16, kernel_size=1)
        self.branch22 = ConvBn(16, 64, kernel_size=3,padding=1)
        self.branch31 = ConvBn(192, 96, kernel_size=1)

    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception3c(nn.Module):
    def __init__(self):
        super(Inception3c, self).__init__()
        self.branch11 = ConvBn(192, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(192, 16, kernel_size=1)
        self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
        self.branch31 = ConvBn(192, 96, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception3d(nn.Module):
    def __init__(self):
        super(Inception3d, self).__init__()
        self.branch11 = ConvBn(192, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(192, 16, kernel_size=1)
        self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
        self.branch31 = ConvBn(192, 96, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception3e(nn.Module):
    def __init__(self):
        super(Inception3e, self).__init__()
        self.branch11 = ConvBn(192, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(192, 16, kernel_size=1)
        self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
        self.branch31 = ConvBn(192, 96, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception4a(nn.Module):
    def __init__(self):
        super(Inception4a, self).__init__()
        self.branch11 = ConvBn(192, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, stride=2, padding=1)
        self.branch21 = ConvBn(192, 32, kernel_size=1)
        self.branch22 = ConvBn(32, 96, kernel_size=3, stride=2, padding=1)
        self.branch31 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch32 = ConvBn(192, 128, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        right = self.branch32(right)
        return torch.cat((left, midle, right), 1)

class Inception4b(nn.Module):
    def __init__(self):
        super(Inception4b, self).__init__()
        self.branch11 = ConvBn(256, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(256, 32, kernel_size=1)
        self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
        self.branch31 = ConvBn(256, 128, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception4c(nn.Module):
    def __init__(self):
        super(Inception4c, self).__init__()
        self.branch11 = ConvBn(256, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(256, 32, kernel_size=1)
        self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
        self.branch31 = ConvBn(256, 128, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception4d(nn.Module):
    def __init__(self):
        super(Inception4d, self).__init__()
        self.branch11 = ConvBn(256, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(256, 32, kernel_size=1)
        self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
        self.branch31 = ConvBn(256, 128, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class Inception4e(nn.Module):
    def __init__(self):
        super(Inception4e, self).__init__()
        self.branch11 = ConvBn(256, 16, kernel_size=1)
        self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
        self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
        self.branch21 = ConvBn(256, 32, kernel_size=1)
        self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
        self.branch31 = ConvBn(256, 128, kernel_size=1)
    def forward(self, input):
        left = self.branch11(input)
        left = self.branch12(left)
        left = self.branch13(left)
        midle = self.branch21(input)
        midle = self.branch22(midle)
        right = self.branch31(input)
        return torch.cat((left, midle, right), 1)

class PVALiteFeat(nn.Module):
    def __init__(self):
        super(PVALiteFeat, self).__init__()
        self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True)
        self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True)
        self.Inception3a = Inception3a()
        self.Inception3b = Inception3b()
        self.Inception3c = Inception3c()
        self.Inception3d = Inception3d()
        self.Inception3e = Inception3e()
        self.Inception4a = Inception4a()
        self.Inception4b = Inception4b()
        self.Inception4c = Inception4c()
        self.Inception4d = Inception4d()
        self.Inception4e = Inception4e()
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.Inception3c(x)
        x = self.Inception3d(x)
        x = self.Inception3e(x)
        x = self.Inception4a(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        x = self.Inception4e(x)
        return x

class liteHyper(PVALiteFeat):
    def __init__(self):
        PVALiteFeat.__init__(self)
        initvars(self.modules())
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

class lite_faster_rcnn(pva_faster_rcnn):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
      self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
      self.dout_base_model = 544
      self.pretrained = pretrained
      self.class_agnostic = class_agnostic
      pva_faster_rcnn.__init__(self, classes, class_agnostic)
      self.rcnn_din = 512
      self.rpn_din = 256

  def _init_modules(self):
    lite = liteHyper()
    #self.pretrained = False
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))

    self.RCNN_base = lite
    
    self.RCNN_cls_score = nn.Linear(512, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(512, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(512, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    #print(pool5_flat.shape)
    fc_features = self.RCNN_top(pool5_flat)
    
    return fc_features