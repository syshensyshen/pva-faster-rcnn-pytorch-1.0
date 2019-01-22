from __future__ import absolute_import
from models.features import pvaHyper
from models.features import liteHyper
from models.lite import PVALiteFeat
import torch
from torchsummary import summary
import cv2
import numpy as np
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = liteHyper().to(device)
model = pvaHyper().to(device)
img_size = 192
num_classes = 1000
batch_size = 12
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
'''
from lib.datasets.pascal_voc import load_pascal_annotation
from lib.datasets.pascal_voc import _get_image_blob
from lib.datasets.pascal_voc import prepareBatchData
root_dir = r'G:\competition\steers\test'
xmllist = []
xmllist.append('0B061732.xml')
xmllist.append('0BF77F48.xml')
xmllist.append('0C006B5C.xml')
xmllist.append('1C318FBA.xml')
gt_boxes, im_blobs, im_scales = prepareBatchData(root_dir, 4, xmllist)
print(gt_boxes.shape)
print(im_blobs.shape)
print(im_scales)
'''

list_1 = [0,1,2,3,4,5,6,7,8]
print(list_1[0:4])