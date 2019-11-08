# -*- coding: utf-8 -*-
'''
# @Author  : syshen
# @date    : 2019/06/17 15:11
# @File    : guided anchors region regression
'''
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.config import cfg
from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from models.smoothl1loss import _smooth_l1_loss
from models.smoothl1loss import _balance_smooth_l1_loss
from lib.rpn.generate_anchors import generate_anchors

import numpy as np
import math
import pdb
import time


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, cfg, din, classes=2):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.num_anchors = len(self.feat_stride)

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        self.feature_adaption = nn.Conv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3)
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels,
                                  1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4,
                                  1)

        # define proposal layer
        self.RPN_proposal = ProposalLayer(
            cfg, self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = torch.from_numpy(generate_anchors(base_size=cfg.FEAT_STRIDE[0], scales=np.array(stride),
                                                         ratios=np.array(1.0))).float()

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, heatmap=None):
        batch_size = base_feat.size(0)
        loc_pred = self.conv_loc(base_feat)
        shape_pred = self.conv_shape(base_feat)
        cls_score = self.conv_cls(base_feat)
        bbox_pred = self.conv_reg(base_feat)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, RPN_bbox_pred.data,
                                  im_info, cfg_key))

        rpn_loss_cls = 0
        rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            self.rpn_loss_box = _balance_smooth_l1_loss(RPN_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                        rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

        return rois, rpn_loss_cls, rpn_loss_box

    def loss(self,
             cls_scores,  # N cls H W
             bbox_preds,  # N 4 H W
             shape_preds,  # N 2 H W
             loc_preds,  # N 1 H W
             gt_bboxes):
