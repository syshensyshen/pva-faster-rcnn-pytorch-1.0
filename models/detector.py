
'''
author: syshen
date: 2019/01/22-01/23
'''
import os
import numpy as np
from collections import OrderedDict
from importlib import import_module

import torch
import torch.nn as nn


class detector(nn.Module):
    def __init__(self, config):
        super(detector, self).__init__()
        self.heads = config.models.structs
        for head in self.heads:
            model = import_module('models.' + head + '.' + self.heads[head]).get_model(config['models'][head])
            self.__setattr__(head, model)

    def forward(self, images, im_info, gt_boxes=None):

        feats = self.__getattr__('backbone')(images)
        feats = self.__getattr__('featurescompose')(feats)
        if self.training:
            rois, rpn_loss_cls, rpn_loss_box = self.__getattr__('rpn_tools')(feats, im_info, gt_boxes)
            rois, cls_score, bbox_pred, rcnn_loss_cls, rcnn_loss_bbox = self.__getattr__('rcnn_tools')(feats, rois, im_info, gt_boxes)
            return rois, cls_score, bbox_pred, rpn_loss_cls, rpn_loss_box, rcnn_loss_cls, rcnn_loss_bbox
        else:
            rois = self.__getattr__('region_tools')(feats, im_info, gt_boxes)
            if isinstance(rois, list) or isinstance(rois, tuple):
                rois = torch.cat(rois)
            rois, cls_score, bbox_pred = self.__getattr__('rcnn')(feats, rois, im_info, gt_boxes)
            return rois, cls_score, bbox_pred