from __future__ import absolute_import
from models.pvanet import PVANetFeat
from models.lite import PVALiteFeat
import torch
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PVALiteFeat().to(device)
summary(model, (3, 224, 224))
#print(model)