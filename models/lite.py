import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
