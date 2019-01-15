from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.config import cfg
#from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from models.smoothl1loss import smooth_l1_loss

import numpy as np
import math
import pdb
import time

class rpn_regression(nn.Module):

    def __init__(self, inchannels):
        super(rpn_regression, self).__init__()

        self.inchannels = inchannels
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.convf_rpn = nn.Conv2d(self.inchannels, 128, 1, 1, 0, bias=True)
        self.convf_2 = nn.Conv2d(self.inchannels, 384, 1, 1, 0, bias=True)
        self.rpn_conv1 = nn.Conv2d(128, 384, 3, 1, 1, bias=True)
        self.rpn_number = len(self.anchor_scales) * len(self.anchor_ratios)
        self.rpn_cls_score = nn.Conv2d(384, self.rpn_number << 1, 1, 1, 0, bias=True)
        self.rpn_bbox_pred = nn.Conv2d(384, self.rpn_number << 2, 1, 1, 0, bias=True)
        self.rpn_anchor_target = AnchorTargetLayer(self.feat_stride, \
                                                   self.anchor_scales,\
                                                   self.anchor_ratios)
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

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

    def forward(self, base_feat, im_info, gt_boxes):
        convf_rpn = F.relu(self.convf_rpn(base_feat))
        convf_2 = F.relu(self.convf_2(base_feat))
        rpn_conv1 = F.relu(self.rpn_conv1(convf_rpn))
        base_feat = torch.cat((convf_rpn, convf_2), 1)
        batch_size = base_feat.size(0)
        rpn_cls_score = self.rpn_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.rpn_number << 1)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv1)
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.rpn_anchor_target.forward((rpn_cls_score.data, gt_boxes, im_info))
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            self.rpn_loss_box = smooth_l1_loss(rpn_bbox_pred, \
                                               rpn_bbox_targets, \
                                               rpn_bbox_inside_weights, \
                                               rpn_bbox_outside_weights, \
                                               sigma=3, dim=[1,2,3])

        return base_feat, rpn_cls_prob, rpn_bbox_pred, \
               self.rpn_loss_cls, self.rpn_loss_box
