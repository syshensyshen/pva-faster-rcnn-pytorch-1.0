from __future__ import absolute_import
from models.features import *
from torchsummary import summary

model = pvaHyper()
summary(model, (3, 224, 224))