import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.pva_faster_rcnn import pva_faster_rcnn
from models.config import cfg

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
        self.Inception3 = nn.Sequential(
            Inception3a(), # 1/2 192
            Inception3b(),
            Inception3c(),
            Inception3d(),
            Inception3e()
        )

        self.Inception4 = nn.Sequential(
            Inception4a(), # 1/2 256
            Inception4b(),
            Inception4c(),
            Inception4d(),
            Inception4e()
        )    
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.Inception3(x)        
        x = self.Inception4(x)
        return x

class liteHyper(PVALiteFeat):
    def __init__(self):
        PVALiteFeat.__init__(self)
        initvars(self.modules())
    def forward(self, input):
        x0 = self.conv1(input) # 1/2 feature
        x1 = self.conv2(x0) # 1/4 feature
        x2 = self.conv3(x1) # 1/8 feature
        x3 = self.Inception3(x2) # 1/16 feature       
        x4 = self.Inception4(x3) # 1/32 feature
        downsample = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)
        upsample = F.interpolate(x4, scale_factor=2, mode="nearest")
        features = torch.cat((downsample, x3, upsample), 1)
        return features

class mCReLU_base(nn.Module):
    def __init__(self, n_in, n_out, kernelsize, stride=1, preAct=False, lastAct=True):
        super(mCReLU_base, self).__init__()
        # Config
        self._preAct = preAct
        self._lastAct = lastAct
        self.act = F.relu

        # Trainable params
        self.conv3x3 = nn.Conv2d(n_in, n_out, kernelsize, stride=stride, padding=int(kernelsize/2))
        self.bn = nn.BatchNorm2d(n_out * 2)

    def forward(self, x):
        if self._preAct:
            x = self.act(x)

        # Conv 3x3 - mCReLU (w/ BN)
        x = self.conv3x3(x)
        x = torch.cat((x, -x), 1)
        x = self.bn(x)

        # TODO: Add scale-bias layer and make 'bn' optional

        if self._lastAct:
            x = self.act(x)

        return x

class mCReLU_residual(nn.Module):
    def __init__(self, n_in, n_red, n_3x3, n_out, kernelsize=3, in_stride=1, proj=False, preAct=False, lastAct=True):
        super(mCReLU_residual, self).__init__()
        # Config
        self._preAct = preAct
        self._lastAct = lastAct
        self._stride = in_stride
        self.act = F.relu

        # Trainable params
        self.reduce = nn.Conv2d(n_in, n_red, 1, stride=in_stride)
        self.conv3x3 = nn.Conv2d(n_red, n_3x3, kernelsize, padding=int(kernelsize/2))
        self.bn = nn.BatchNorm2d(n_3x3 * 2)
        self.expand = nn.Conv2d(n_3x3 * 2, n_out, 1)

        if in_stride > 1:
            # TODO: remove this assertion
            assert(proj)

        self.proj = nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None

    def forward(self, x):
        x_sc = x

        if self._preAct:
            x = self.act(x)

        # Conv 1x1 - Relu
        x = self.reduce(x)
        x = self.act(x)

        # Conv 3x3 - mCReLU (w/ BN)
        x = self.conv3x3(x)
        x = torch.cat((x, -x), 1)
        x = self.bn(x)
        x = self.act(x)

        # TODO: Add scale-bias layer and make 'bn' optional

        # Conv 1x1
        x = self.expand(x)

        if self._lastAct:
            x = self.act(x)

        # Projection
        if self.proj:
            x_sc = self.proj(x_sc)

        x = x + x_sc

        return x

class shortlitehyper(nn.Module):
    def __init__(self, pretrained=False):
        super(shortlitehyper, self).__init__()
        self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True) # 1/2
        self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True) # 1/4 48
        self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True) # 1/8 96
        self.Inception3 = nn.Sequential(
            Inception3a(), # 1/16 192
            Inception3b(),
            Inception3c(),
            Inception3d(),
            Inception3e()
        )

        self.Inception4 = nn.Sequential(
            Inception4a(), # 1/32 256
            Inception4b(),
            Inception4c(),
            Inception4d(),
            Inception4e()
        )
        #self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)
        #self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.align = nn.Conv2d(256, 192, kernel_size=1, stride=1)
        initvars(self.modules())
    def forward(self, input):
        #print(input.shape)       
        x0 = self.conv1(input) # 1/2 feature 32
        x1 = self.conv2(x0) # 1/4 feature 48
        x2 = self.conv3(x1) # 1/8 feature 96
        x3 = self.Inception3(x2) # 1/16 feature 192
        x4 = self.Inception4(x3) # 1/32 feature 256
        downsample1 = self.downsample1(x1)
        upsample1 = F.interpolate(x4, scale_factor=2, mode="nearest")
        upsample1 = self.align(upsample1) # 192
        x3 = x3 + upsample1 # 192
        upsample2 = F.interpolate(x3, scale_factor=2, mode="nearest") # 192
        features = torch.cat((downsample1, x2, upsample2), 1)
        return features # 336

class shortshortlitehyper(nn.Module):
    def __init__(self, pretrained=False):
        super(shortshortlitehyper, self).__init__()
        self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True) # 1/2
        self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True) # 1/4 48
        self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True) # 1/8 96
        self.Inception3 = nn.Sequential(
            Inception3a(), # 1/16 192
            Inception3b(),
            Inception3c(),
            Inception3d(),
            Inception3e()
        )

        self.Inception4 = nn.Sequential(
            Inception4a(), # 1/32 256
            Inception4b(),
            Inception4c(),
            Inception4d(),
            Inception4e()
        )
        self.downsample1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.align1 = nn.Conv2d(256, 192, kernel_size=1, stride=1)
        self.align2 = nn.Conv2d(192, 96, kernel_size=1, stride=1)
        initvars(self.modules())
    def forward(self, input):
        #print(input.shape)
        x0 = self.conv1(input) # 1/2 feature 32
        x1 = self.conv2(x0) # 1/4 feature 48
        x2 = self.conv3(x1) # 1/8 feature 96
        x3 = self.Inception3(x2) # 1/16 feature 192
        x4 = self.Inception4(x3) # 1/32 feature 256

        downsample1 = self.downsample1(x0)
        upsample1 = F.interpolate(x4, scale_factor=2, mode="nearest")
        x3 = x3 + self.align1(upsample1) # 256->192
        upsample1 = F.interpolate(x3, scale_factor=2, mode="nearest") # 192
        x2 = x2 + self.align2(upsample1) # 192->96
        upsample2 = F.interpolate(x2, scale_factor=2, mode="nearest") # 96
        features = torch.cat((downsample1, x1, upsample2), 1)
        return features # 32+48+96=176

class lite_faster_rcnn(pva_faster_rcnn):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
      #self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
      self.dout_base_model = int(cfg.MODEL.DOUT_BASE_MODEL)
      self.pretrained = pretrained
      self.class_agnostic = class_agnostic
      #self.rcnn_din = 512
      #self.rpn_din = 256
      pva_faster_rcnn.__init__(self, classes, class_agnostic)      

  def _init_modules(self):

    self.pretrained = False

    self.RCNN_base = eval(cfg.MODEL.BACKBONE)()
    
    self.RCNN_cls_score = nn.Linear(self.rcnn_last_din, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(self.rcnn_last_din, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(self.rcnn_last_din, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    #print(pool5_flat.shape)
    fc_features = self.RCNN_top(pool5_flat)
    
    return fc_features
