from __future__ import absolute_import
import torch
from torchsummary import summary
import cv2
import numpy as np
from models.features import pvaHyper
from models.features import liteHyper
from models.lite import PVALiteFeat
from lib.rpn.rpn_regression import rpn_regression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
litehyper = liteHyper().to(device)
pvahyper = pvaHyper().to(device)
img_size = 192
num_classes = 1000
batch_size = 12
#model = PVALiteNet(inputsize=img_size, num_classes = num_classes).to(device)
#input = torch.empty(batch_size, 3, img_size, img_size, dtype=torch.float32).to(device)
#features = model.forward(input)
#print(features)
#print(features.shape)
#summary(model, (3, 320, 320))
#print(model)

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
from lib.datasets.pascal_voc import prepareBatchData
root_dir = r'E:\VOCdevkit\VOC2007\data'
xmllist = []
xmllist.append('000001.xml')
xmllist.append('000002.xml')
xmllist.append('000003.xml')
xmllist.append('000004.xml')
gt_boxes, im_blobs, im_scales = prepareBatchData(root_dir, 4, xmllist)
gt_boxes = torch.from_numpy(gt_boxes) 
im_blobs = torch.from_numpy(im_blobs)
im_scales = torch.from_numpy(im_scales)
features = litehyper.forward(im_blobs.to(device))
print(features.shape)
#inchannles = features.shape(1)
rpn_regression = rpn_regression(544).to(device)
base_feat, rpn_cls_prob, rpn_bbox_pred, rpn_loss_cls, rpn_loss_box = \
rpn_regression.forward(features.to(device), im_scales.to(device), gt_boxes.to(device))
print(base_feat.shape)
print(rpn_cls_prob.shape)
print(rpn_bbox_pred.shape)
print(rpn_loss_cls)
print(rpn_loss_box)