from __future__ import absolute_import
from models.features import pvaHyper
from models.features import liteHyper
from models.lite import PVALiteNet
import torch
from torchsummary import summary
import cv2
import numpy as np
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = liteHyper().to(device)
#model = pvaHyper().to(device)
img_size = 192
num_classes = 1000
batch_size = 12
model = PVALiteNet(inputsize=img_size, num_classes = num_classes).to(device)
input = torch.empty(batch_size, 3, img_size, img_size, dtype=torch.float32).to(device)
features = model.forward(input)
print(features)
print(features.shape)
#summary(model, (3, 320, 320))
#print(model)
'''
'''
_feat_stride = 16
width = 40
height = 40
import numpy as np
shift_x = np.arange(0, width) * _feat_stride
shift_y = np.arange(0, height) * _feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
print(shift_x.shape)
print(shift_y.shape)
import torch
shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
#shifts = shifts.contiguous().type_as(rpn_cls_score).float()
print(shifts.shape)
from lib.rpn.generate_anchors import generate_anchors

# Anchor scales for RPN
scales = [8,16,32]
# Anchor ratios for RPN
ratios = [0.5,1,2]
anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()
print(anchors.size(0))
a=anchors.new(4, anchors.size(0))
print(a.shape)
batch_size = 4
N = 9
banchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
print(banchors.shape)
'''

from lib.datasets.pascal_voc import load_pascal_annotation
from lib.datasets.pascal_voc import _get_image_blob
xml_context = load_pascal_annotation(r'E:\VOCdevkit\VOC2007\Annotations\000001.xml')
print(xml_context['boxes'].shape[0])
test_gt = np.zeros((2, 4, 5), dtype=np.float32)
print(test_gt[0,0:2,0:3])
img = cv2.imread(r"E:\VOCdevkit\VOC2007\JPEGImages\000001.jpg")
print(img.shape)
blob, im_scale = _get_image_blob(img, np.array([[[102.9801, 115.9465, 122.7717]]]), 512, 1440, 32)
print(blob.shape, im_scale)
cv2.imshow('img', img)
cv2.waitKey(0)