# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# from models.pva_faster_rcnn import pva_faster_rcnn
# #from models.config import cfg
# import torch.utils.model_zoo as model_zoo
#
# def initvars(modules):
#     # Copied from vision/torchvision/models/resnet.py
#     for m in modules:
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#
# class ConvBn(nn.Module):
#     def __init__(self, in_feature, out_feature, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(ConvBn, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size,
#                       stride=stride, padding=padding, dilation=dilation,
#                       groups=groups, bias=bias),
#             nn.BatchNorm2d(out_feature),
#             nn.ReLU(inplace=True))
#     def forward(self, input):
#         return self.conv(input)
#
# class Inception3a(nn.Module):
#     def __init__(self):
#         super(Inception3a, self).__init__()
#         self.branch11 = ConvBn(96, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, stride=2, padding=1)
#         self.branch21 = ConvBn(96, 16, kernel_size=1)
#         self.branch22 = ConvBn(16, 64, kernel_size=3, stride=2, padding=1)
#         self.branch31 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.branch32 = ConvBn(96, 96, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         right = self.branch32(right)
#         return torch.cat((left, midle, right), 1)
#
# class Inception3b(nn.Module):
#     def __init__(self):
#         super(Inception3b, self).__init__()
#         self.branch11 = ConvBn(192, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(192, 16, kernel_size=1)
#         self.branch22 = ConvBn(16, 64, kernel_size=3,padding=1)
#         self.branch31 = ConvBn(192, 96, kernel_size=1)
#
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception3c(nn.Module):
#     def __init__(self):
#         super(Inception3c, self).__init__()
#         self.branch11 = ConvBn(192, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(192, 16, kernel_size=1)
#         self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(192, 96, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception3d(nn.Module):
#     def __init__(self):
#         super(Inception3d, self).__init__()
#         self.branch11 = ConvBn(192, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(192, 16, kernel_size=1)
#         self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(192, 96, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception3e(nn.Module):
#     def __init__(self):
#         super(Inception3e, self).__init__()
#         self.branch11 = ConvBn(192, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(192, 16, kernel_size=1)
#         self.branch22 = ConvBn(16, 64, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(192, 96, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception4a(nn.Module):
#     def __init__(self):
#         super(Inception4a, self).__init__()
#         self.branch11 = ConvBn(192, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, stride=2, padding=1)
#         self.branch21 = ConvBn(192, 32, kernel_size=1)
#         self.branch22 = ConvBn(32, 96, kernel_size=3, stride=2, padding=1)
#         self.branch31 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.branch32 = ConvBn(192, 128, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         right = self.branch32(right)
#         return torch.cat((left, midle, right), 1)
#
# class Inception4b(nn.Module):
#     def __init__(self):
#         super(Inception4b, self).__init__()
#         self.branch11 = ConvBn(256, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(256, 32, kernel_size=1)
#         self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(256, 128, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception4c(nn.Module):
#     def __init__(self):
#         super(Inception4c, self).__init__()
#         self.branch11 = ConvBn(256, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(256, 32, kernel_size=1)
#         self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(256, 128, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception4d(nn.Module):
#     def __init__(self):
#         super(Inception4d, self).__init__()
#         self.branch11 = ConvBn(256, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(256, 32, kernel_size=1)
#         self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(256, 128, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class Inception4e(nn.Module):
#     def __init__(self):
#         super(Inception4e, self).__init__()
#         self.branch11 = ConvBn(256, 16, kernel_size=1)
#         self.branch12 = ConvBn(16, 32, kernel_size=3, padding=1)
#         self.branch13 = ConvBn(32, 32, kernel_size=3, padding=1)
#         self.branch21 = ConvBn(256, 32, kernel_size=1)
#         self.branch22 = ConvBn(32, 96, kernel_size=3, padding=1)
#         self.branch31 = ConvBn(256, 128, kernel_size=1)
#     def forward(self, input):
#         left = self.branch11(input)
#         left = self.branch12(left)
#         left = self.branch13(left)
#         midle = self.branch21(input)
#         midle = self.branch22(midle)
#         right = self.branch31(input)
#         return torch.cat((left, midle, right), 1)
#
# class PVALiteFeat(nn.Module):
#     def __init__(self):
#         super(PVALiteFeat, self).__init__()
#         self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True)
#         self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True)
#         self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True)
#         self.Inception3 = nn.Sequential(
#             Inception3a(), # 1/2 192
#             Inception3b(),
#             Inception3c(),
#             Inception3d(),
#             Inception3e()
#         )
#
#         self.Inception4 = nn.Sequential(
#             Inception4a(), # 1/2 256
#             Inception4b(),
#             Inception4c(),
#             Inception4d(),
#             Inception4e()
#         )
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.Inception3(x)
#         x = self.Inception4(x)
#         return x
#
# class liteHyper(PVALiteFeat):
#     def __init__(self):
#         PVALiteFeat.__init__(self)
#         initvars(self.modules())
#     def forward(self, input):
#         x0 = self.conv1(input) # 1/2 feature
#         x1 = self.conv2(x0) # 1/4 feature
#         x2 = self.conv3(x1) # 1/8 feature
#         x3 = self.Inception3(x2) # 1/16 feature
#         x4 = self.Inception4(x3) # 1/32 feature
#         downsample = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)
#         upsample = F.interpolate(x4, scale_factor=2, mode="nearest")
#         features = torch.cat((downsample, x3, upsample), 1)
#         return features
#
# class mCReLU_base(nn.Module):
#     def __init__(self, n_in, n_out, kernelsize, stride=1, preAct=False, lastAct=True):
#         super(mCReLU_base, self).__init__()
#         # Config
#         self._preAct = preAct
#         self._lastAct = lastAct
#         self.act = F.relu
#
#         # Trainable params
#         self.conv3x3 = nn.Conv2d(n_in, n_out, kernelsize, stride=stride, padding=int(kernelsize/2))
#         self.bn = nn.BatchNorm2d(n_out * 2)
#
#     def forward(self, x):
#         if self._preAct:
#             x = self.act(x)
#
#         # Conv 3x3 - mCReLU (w/ BN)
#         x = self.conv3x3(x)
#         x = torch.cat((x, -x), 1)
#         x = self.bn(x)
#
#         # TODO: Add scale-bias layer and make 'bn' optional
#
#         if self._lastAct:
#             x = self.act(x)
#
#         return x
#
# class mCReLU_residual(nn.Module):
#     def __init__(self, n_in, n_red, n_3x3, n_out, kernelsize=3, in_stride=1, proj=False, preAct=False, lastAct=True):
#         super(mCReLU_residual, self).__init__()
#         # Config
#         self._preAct = preAct
#         self._lastAct = lastAct
#         self._stride = in_stride
#         self.act = F.relu
#
#         # Trainable params
#         self.reduce = nn.Conv2d(n_in, n_red, 1, stride=in_stride)
#         self.conv3x3 = nn.Conv2d(n_red, n_3x3, kernelsize, padding=int(kernelsize/2))
#         self.bn = nn.BatchNorm2d(n_3x3 * 2)
#         self.expand = nn.Conv2d(n_3x3 * 2, n_out, 1)
#
#         if in_stride > 1:
#             # TODO: remove this assertion
#             assert(proj)
#
#         self.proj = nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None
#
#     def forward(self, x):
#         x_sc = x
#
#         if self._preAct:
#             x = self.act(x)
#
#         # Conv 1x1 - Relu
#         x = self.reduce(x)
#         x = self.act(x)
#
#         # Conv 3x3 - mCReLU (w/ BN)
#         x = self.conv3x3(x)
#         x = torch.cat((x, -x), 1)
#         x = self.bn(x)
#         x = self.act(x)
#
#         # TODO: Add scale-bias layer and make 'bn' optional
#
#         # Conv 1x1
#         x = self.expand(x)
#
#         if self._lastAct:
#             x = self.act(x)
#
#         # Projection
#         if self.proj:
#             x_sc = self.proj(x_sc)
#
#         x = x + x_sc
#
#         return x
#
# class shortlitehyper(nn.Module):
#     def __init__(self, pretrained=False):
#         super(shortlitehyper, self).__init__()
#         self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True) # 1/2
#         self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True) # 1/4 48
#         self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True) # 1/8 96
#         self.Inception3 = nn.Sequential(
#             Inception3a(), # 1/16 192
#             Inception3b(),
#             Inception3c(),
#             Inception3d(),
#             Inception3e()
#         )
#
#         self.Inception4 = nn.Sequential(
#             Inception4a(), # 1/32 256
#             Inception4b(),
#             Inception4c(),
#             Inception4d(),
#             Inception4e()
#         )
#         #self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.downsample1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)
#         #self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.align = nn.Conv2d(256, 192, kernel_size=1, stride=1)
#         initvars(self.modules())
#     def forward(self, input):
#         #print(input.shape)
#         x0 = self.conv1(input) # 1/2 feature 32
#         x1 = self.conv2(x0) # 1/4 feature 48
#         x2 = self.conv3(x1) # 1/8 feature 96
#         x3 = self.Inception3(x2) # 1/16 feature 192
#         x4 = self.Inception4(x3) # 1/32 feature 256
#         downsample1 = self.downsample1(x1)
#         upsample1 = F.interpolate(x4, scale_factor=2, mode="nearest")
#         upsample1 = self.align(upsample1) # 192
#         x3 = x3 + upsample1 # 192
#         upsample2 = F.interpolate(x3, scale_factor=2, mode="nearest") # 192
#         features = torch.cat((downsample1, x2, upsample2), 1)
#         return features # 336
#
# class shortshortlitehyper(nn.Module):
#     def __init__(self, pretrained=False):
#         super(shortshortlitehyper, self).__init__()
#         self.conv1 = ConvBn(3, 32, kernel_size=3, stride=2, bias=True) # 1/2
#         self.conv2 = ConvBn(32, 48, kernel_size=3, stride=2, padding=1, bias=True) # 1/4 48
#         self.conv3 = ConvBn(48, 96, kernel_size=3, stride=2, padding=1, bias=True) # 1/8 96
#         self.Inception3 = nn.Sequential(
#             Inception3a(), # 1/16 192
#             Inception3b(),
#             Inception3c(),
#             Inception3d(),
#             Inception3e()
#         )
#
#         self.Inception4 = nn.Sequential(
#             Inception4a(), # 1/32 256
#             Inception4b(),
#             Inception4c(),
#             Inception4d(),
#             Inception4e()
#         )
#         self.downsample1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.align1 = nn.Conv2d(256, 192, kernel_size=1, stride=1)
#         self.align2 = nn.Conv2d(192, 96, kernel_size=1, stride=1)
#         initvars(self.modules())
#     def forward(self, input):
#         #print(input.shape)
#         x0 = self.conv1(input) # 1/2 feature 32
#         x1 = self.conv2(x0) # 1/4 feature 48
#         x2 = self.conv3(x1) # 1/8 feature 96
#         x3 = self.Inception3(x2) # 1/16 feature 192
#         x4 = self.Inception4(x3) # 1/32 feature 256
#
#         downsample1 = self.downsample1(x0)
#         upsample1 = F.interpolate(x4, scale_factor=2, mode="nearest")
#         x3 = x3 + self.align1(upsample1) # 256->192
#         upsample1 = F.interpolate(x3, scale_factor=2, mode="nearest") # 192
#         x2 = x2 + self.align2(upsample1) # 192->96
#         upsample2 = F.interpolate(x2, scale_factor=2, mode="nearest") # 96
#         features = torch.cat((downsample1, x1, upsample2), 1)
#         return features # 32+48+96=176
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class ResnetHyper(nn.Module):
#
#      '''
#      hyper dim: 0->stride == 4
#                1->stride == 8
#                2->stride == 16
#                3->stride == 32
#      '''
#     def __init__(self, block, layers, hyper_dim=2):
#         self.inplanes = 64
#         super(ResnetHyper, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         if block == BasicBlock:
#             self.fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
#                          self.layer4[layers[3] - 1].conv2.out_channels]
#         elif block == Bottleneck:
#             self.fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
#                          self.layer4[layers[3] - 1].conv3.out_channels]
#
#         self.downsample1 = nn.Conv2d(self.fpn_sizes[0], self.fpn_sizes[0], kernel_size=1, stride=2, padding=0)
#         self.align1 = nn.Conv2d(self.fpn_sizes[3], self.fpn_sizes[2], kernel_size=1, stride=1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#         self.freeze_bn()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def freeze_bn(self):
#         '''Freeze BatchNorm layers.'''
#         for layer in self.modules():
#             if isinstance(layer, nn.BatchNorm2d):
#                 layer.eval()
#
#     def forward(self, img_batch):
#         x = self.conv1(img_batch) # 1/2
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x) # 1/4
#
#         x1 = self.layer1(x) # 1/4
#         x2 = self.layer2(x1) # 1/8
#         x3 = self.layer3(x2) # 1/16
#         x4 = self.layer4(x3) # 1/32
#
#         downsample1 = self.downsample1(x1)
#         upsample1 = F.interpolate(x4, scale_factor=2, mode="nearest")
#         x3 = x3 + self.align1(upsample1)  # 256->192
#         upsample1 = F.interpolate(x3, scale_factor=2, mode="nearest")
#         features = torch.cat((downsample1, x2, upsample1), 1)
#         return features  # 32+48+96=176
#
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResnetHyper(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='/home/user/.torch/models/'),
#                               strict=False)
#     return model
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResnetHyper(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='/home/user/.torch/models/'),
#                               strict=False)
#     return model
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResnetHyper(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='/home/user/.torch/models/'),
#                               strict=False)
#     return model
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResnetHyper(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='/home/user/.torch/models/'),
#                               strict=False)
#     return model
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResnetHyper(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='/home/user/.torch/models/'),
#                               strict=False)
#     return model
#
# class lite_faster_rcnn(pva_faster_rcnn):
#   def __init__(self, cfg, classes, pretrained=False, class_agnostic=False):
#       #self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
#       self.dout_base_model = int(cfg.MODEL.DOUT_BASE_MODEL)
#       self.pretrained = pretrained
#       self.class_agnostic = class_agnostic
#       #self.rcnn_din = 512
#       #self.rpn_din = 256
#       pva_faster_rcnn.__init__(self, cfg, classes, class_agnostic)
#
#   def _init_modules(self, cfg):
#
#     self.pretrained = False
#
#     self.backbone = eval(cfg.MODEL.BACKBONE)()
#
#     self.rcnn_cls_score = nn.Linear(self.rcnn_last_din, self.n_classes)
#
#     if self.class_agnostic:
#       self.rcnn_bbox_pred = nn.Linear(self.rcnn_last_din, 4)
#     else:
#       self.rcnn_bbox_pred = nn.Linear(self.rcnn_last_din, 4 * self.n_classes)
#
#   def _head_to_tail(self, pool5):
#
#     pool5_flat = pool5.view(pool5.size(0), -1)
#     #print(pool5_flat.shape)
#     fc_features = self.rcnn_top(pool5_flat)
#
#     return fc_features
