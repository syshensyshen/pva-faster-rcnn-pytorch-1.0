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
from torch.autograd import Variable

# from models.config import cfg
from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from models.smoothl1loss import _smooth_l1_loss, _smooth_l1_loss_by_zcc
from models.smoothl1loss import _balance_smooth_l1_loss

import numpy as np
import math
import pdb
import time

# added by Henson
from models.giou import compute_iou


class _RPN(nn.Module):

    """ region proposal network """

    def __init__(self, cfg, din, rpn_din):
        super(_RPN, self).__init__()

        self.cfg = cfg

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.rpn_din = rpn_din

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, self.rpn_din, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 2(bg/fg) * 9 (anchors)
        self.nc_score_out = len(self.anchor_scales) * \
            len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(
            self.rpn_din, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 4(coords) * 9 (anchors)
        self.nc_bbox_out = len(self.anchor_scales) * \
            len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(
            self.rpn_din, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = ProposalLayer(
            cfg, self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = AnchorTargetLayer(
            cfg, self.feat_stride, self.anchor_scales, self.anchor_ratios)

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
        cfg = self.cfg

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        RPN_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(RPN_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        RPN_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois, all_proposals = self.RPN_proposal((rpn_cls_prob.data, RPN_bbox_pred.data,
                                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target(
                (RPN_cls_score.data, gt_boxes, im_info))

            # compute classification loss
            RPN_cls_score = rpn_cls_score_reshape.permute(
                0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            RPN_cls_score = torch.index_select(
                RPN_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(
                rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            if self.cfg.TRAIN.is_ohem_rpn:  # added by Henson
                rpn_loss_cls = F.cross_entropy(
                    RPN_cls_score, rpn_label, reduction='none')

                top_k = int(0.125 * self.cfg.TRAIN.RPN_BATCHSIZE *
                            base_feat.size(0))
                _, topk_loss_inds = rpn_loss_cls.topk(top_k)
                self.rpn_loss_cls = rpn_loss_cls[topk_loss_inds].mean()
            else:
                self.rpn_loss_cls = F.cross_entropy(RPN_cls_score, rpn_label)

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[
                1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            if cfg.TRAIN.loss_type == "smoothL1loss":
                if self.cfg.TRAIN.is_ohem_rpn:  # added by Henson
                    rpn_loss_box = _smooth_l1_loss_by_zcc(RPN_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                          rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
                    rpn_loss_box = rpn_loss_box.view(-1)

                    top_k = int(0.125 * rpn_bbox_inside_weights.sum() + 0.5)
                    # print("=> top_k: ", top_k)
                    _, topk_loss_inds = rpn_loss_box.topk(top_k)
                    self.rpn_loss_box = rpn_loss_box[topk_loss_inds].mean()

                else:
                    self.rpn_loss_box = _smooth_l1_loss(RPN_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                        rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

                # self.rpn_loss_box = _balance_smooth_l1_loss(RPN_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                #                                                 rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

                # iou, g_iou = compute_iou(
                #     gt_boxes[:, :, 0:4].view(-1, 4), gt_boxes[:, :, 0:4].view(-1, 4))  # all_proposals.view(-1, 4)

            elif "IOUloss" in cfg.TRAIN.loss_type:
                iou, g_iou = compute_iou(
                    rpn_bbox_targets, rpn_bbox_targets, rpn_bbox_inside_weights,
                    rpn_bbox_outside_weights)
                if cfg.TRAIN.loss_type == "GIOUloss":
                    self.rpn_loss_box = 1 - g_iou
                elif cfg.TRAIN.loss_type == "IOUloss":
                    self.rpn_loss_box = -iou.log()

        return rois, self.rpn_loss_cls, self.rpn_loss_box
