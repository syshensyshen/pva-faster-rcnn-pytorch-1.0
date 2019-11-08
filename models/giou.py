import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(output, target, bbox_inside_weights=None, bbox_outside_weights=None,
                transform_weights=None, batch_size=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    if batch_size is None:
        batch_size = output.size(0)

    x1, y1, x2, y2 = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    x1g, y1g, x2g, y2g = target[:, 0], target[:,
                                              1], target[:, 2], target[:, 3],

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)

    # iou_weights = bbox_inside_weights.view(-1, 4).mean(
    #     1)*bbox_outside_weights.view(-1, 4).mean(1)
    # iou_weights = 1.0

    # iouk = ((1 - iouk) * iou_weights).sum(0) / batch_size
    # miouk = ((1 - miouk) * iou_weights).sum(0) / batch_size

    return iouk, miouk
