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

# from models.config import cfg
from .proposal_target_layer import ProposalTargetLayer
from models.smoothl1loss import _smooth_l1_loss
from collections import OrderedDict
from torchvision.ops import RoIPool as ROIPoolingLayer
from torchvision.ops import RoIAlign as ROIAlignLayer
from ..backbone.basic_modules import kaiming_init

class RCNNModule(nn.Module):
    def __init__(self, cfg):
        super(RCNNModule, self).__init__()

        stride = cfg.stride
        inchannels = cfg.inchannels
        pooling_size = cfg.pooling_size
        rcnn_first = cfg.rcnn_first
        rcnn_last_din = cfg.rcnn_last_din
        mode = cfg.mode
        mean = cfg.mean
        std = cfg.std
        inside_weight = cfg.inside_weight
        fraction = cfg.fraction
        batch_size = cfg.batch_size
        fg_thresh = cfg.fg_thresh
        bg_thresh_hi = cfg.bg_thresh_hi
        bg_thresh_lo = cfg.bg_thresh_lo
        self.with_avg_pool = cfg.with_avg_pool
        self.n_classes = len(cfg.class_list[0]) + 1
        self.class_agnostic = cfg.class_agnostic
        self.ohem_rcnn = cfg.ohem_rcnn

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.pooling_size)
        self.proposal_target = ProposalTargetLayer(mean, std, inside_weight, batch_size, fraction, self.n_classes,
                                                   fg_thresh, bg_thresh_hi, bg_thresh_lo)

        if mode == 'align':
            self.roi_pool = ROIAlignLayer(
                (pooling_size, pooling_size), 1.0 / stride, 2)
        elif mode == 'pool':
            self.roi_pool = ROIPoolingLayer(
                (pooling_size, pooling_size), 1.0 / stride)
        else:
            print('none mode')
            exit(-1)

        self.pre_regression = nn.Sequential(OrderedDict([
            ('fc6_', nn.Linear(inchannels * pooling_size * pooling_size, rcnn_first)),
            ('fc6_r', nn.ReLU(inplace=True)),
            ('fc7_', nn.Linear(rcnn_first, rcnn_last_din, bias=True)),
            ('fc7_r', nn.ReLU(inplace=True))
        ]))

        self.cls_score = nn.Linear(rcnn_last_din, self.n_classes)

        if self.class_agnostic:
            self.bbox_pred = nn.Linear(rcnn_last_din, 4)
        else:
            self.bbox_pred = nn.Linear(rcnn_last_din, 4 * self.n_classes)

        self.__init_weights__()

    def __init_weights__(self):
        kaiming_init(self.pre_regression.fc6_)
        kaiming_init(self.pre_regression.fc7_)
        kaiming_init(self.cls_score)
        kaiming_init(self.bbox_pred)


    def forward(self, feats, rois, im_info, gt_boxes):

        batch_size = feats.size(0)
        with torch.no_grad():
            if self.training:
                roi_data = self.proposal_target(rois, gt_boxes)
                rois, labels, bbox_target, inside_w, outside_w = roi_data

                labels = labels.view(-1).long()
                bbox_target = bbox_target.view(-1, bbox_target.size(2))
                inside_w = inside_w.view(-1, inside_w.size(2))
                outside_w =  outside_w.view(-1, outside_w.size(2))
            else:
                labels = None
                bbox_target = None
                inside_w = None
                outside_w = None

        roi_feats = self.roi_pool(feats, rois.reshape(-1, 5))
        if self.with_avg_pool:
            roi_feats = self.avg_pool(roi_feats)
        else:
            roi_feats = roi_feats.reshape(roi_feats.size(0), -1)

        roi_feats = self.pre_regression(roi_feats)
        bbox_pred = self.bbox_pred(roi_feats)
        cls_score = self.cls_score(roi_feats)
        cls_prob = F.softmax(cls_score, 1)

        if self.training:
            # classification loss
            if self.ohem_rcnn:
                loss_cls = F.cross_entropy(cls_score, labels, reduction='none')
                top_k = self.topk
                _, topk_loss_inds = loss_cls.topk(top_k)
                loss_cls = loss_cls[topk_loss_inds].mean()
                loss_bbox = _smooth_l1_loss(bbox_pred[topk_loss_inds, :], bbox_target[topk_loss_inds, :],
                                                 inside_w[topk_loss_inds, :], outside_w[topk_loss_inds, :],
                                                 sigma=3.0)
            else:
                loss_cls = F.cross_entropy(cls_score, labels)
                loss_bbox = _smooth_l1_loss(bbox_pred, bbox_target, inside_w, outside_w)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, bbox_pred, loss_cls, loss_bbox
        else:
            return rois, cls_prob, bbox_pred


def get_model(config):
    return RCNNModule(config)