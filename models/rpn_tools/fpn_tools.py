# -*- coding: utf-8 -*-
'''
# @Author  : syshen
# @date    : 2019/1/16 18:19
# @File    : region regression
'''
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proposal_layer import ProposalLayer
from .anchor_target_layer import AnchorTargetLayer
from models.smoothl1loss import _smooth_l1_loss

import numpy as np

from ..backbone.basic_modules import kaiming_init, normal_init

from models.rpn_tools.generate_anchors import generate_anchors


class Anchor(object):

    def __init__(self, stride, ratios, scales):

        self.stride = stride
        self.base_anchor = torch.from_numpy(generate_anchors(base_size=stride, scales=np.array(scales), ratios=np.array(ratios))).float()

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def __call__(self, feat_height, feat_width, device):
        base_anchors = self.base_anchor.to(device)
        shift_x = torch.arange(0, feat_width, device=device) * self.stride
        shift_y = torch.arange(0, feat_height, device=device) * self.stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)

        return all_anchors

class RPNModule(nn.Module):

    """ region proposal network """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.inchannels = cfg.inchannels  # get depth of input feature map, e.g., 512
        self.output_channels = cfg.output_channels  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.scales
        self.anchor_ratios = cfg.ratios
        self.feat_stride = cfg.stride

        # proposal layer
        self.pre_nms = cfg.pre_nms
        self.post_nms = cfg.post_nms
        # self.nms_thresh = cfg.model.rpn.nms_thresh

        rpn_batch = cfg.batch_size
        negative_overlap = cfg.neg_thresh
        positive_overlap = cfg.pos_thresh
        positive_weight = cfg.pos_weight
        fraction = cfg.fraction
        inside_weight = cfg.inside_weight
        clobber_positive = cfg.clobber_positive
        self.nms_thresh = cfg.nms_thresh

        # define the convrelu layers processing input feature map
        self.rpn_preconv = nn.Conv2d(self.inchannels, self.output_channels, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 2(bg/fg) * 9 (anchors)
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        self.nc_score_out = self.num_anchors * 2
        self.cls_pred = nn.Conv2d( self.output_channels, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 4(coords) * 9 (anchors)
        self.nc_bbox_out = self.nc_score_out * 2
        self.bbox_pred = nn.Conv2d(self.output_channels, self.nc_bbox_out, 1, 1, 0)

        # define anchor genarator layer
        self.anchor = Anchor(self.feat_stride, self.anchor_ratios, self.anchor_scales)

        # define proposal layer
        self.proposals = ProposalLayer( self.feat_stride, nms_thresh=self.nms_thresh, min_size=self.feat_stride)

        # define anchor target layer
        self.get_targets = AnchorTargetLayer(rpn_batch, negative_overlap, positive_overlap, positive_weight, fraction, inside_weight, clobber_positive)

        # self.rpn_loss_cls = 0
        # self.rpn_loss_box = 0
    def __init_weights__(self):
        kaiming_init(self.cls_pred)
        kaiming_init(self.bbox_pred)

    def rpn_loss(self, scores, bboxes_pred, anchors, im_info, gt_boxes, loss_type="smoothL1loss"):

        batch_size,_,height,width = scores.size()

        with torch.no_grad():
            rpn_label, bbox_targets, inside_weights, outside_weights = \
                self.get_targets(scores.detach(), anchors, self.num_anchors, im_info, gt_boxes)

        # compute classification loss
        scores = scores.reshape(batch_size, 2, -1, width)
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        rpn_label = rpn_label.reshape(batch_size, -1)
        rpn_keep = rpn_label.reshape(-1).ne(-1).nonzero().reshape(-1)
        scores = torch.index_select(scores.reshape(-1, 2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.reshape(-1), 0, rpn_keep.data).long()

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_loss_cls = F.cross_entropy(scores, rpn_label)

        rpn_loss_box = _smooth_l1_loss(bboxes_pred, bbox_targets, inside_weights, outside_weights, sigma=3, dim=[1, 2, 3])

        return rpn_loss_cls, rpn_loss_box

    def forward(self, base_feat, im_info, gt_boxes):

        batch_size, _ , feat_height, feat_width = base_feat.size()

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.rpn_preconv(base_feat), inplace=True)
        # get rpn classification score
        scores = self.cls_pred(rpn_conv1)

        # get rpn offsets to the anchor boxes
        bboxes_pred = self.bbox_pred(rpn_conv1)

        anchors = self.anchor(feat_height, feat_width, base_feat.device)

        rois, all_proposals = self.proposals(scores.detach(), bboxes_pred.detach(), anchors, self.num_anchors, im_info, self.pre_nms, self.post_nms, self.nms_thresh)

        # generating training labels and build the rpn loss
        rpn_loss_cls = rpn_loss_box = 0.0

        if self.training:
            assert gt_boxes is not None

            rpn_loss_cls, rpn_loss_box = self.rpn_loss(
                scores, bboxes_pred, anchors, im_info, gt_boxes
            )
            return rois, rpn_loss_cls, rpn_loss_box
        else:
            return rois


def get_model(config):
    return RPNSingleModule(config)