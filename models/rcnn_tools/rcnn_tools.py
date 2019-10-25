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

class RCNNSingleModule(nn.Module):
    def __init__(self, cfg):
        super(RCNNSingleModule, self).__init__()

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
                bbox_target = bbox_target.reshape(-1, bbox_target.size(2))
                inside_w = inside_w.reshape(-1, inside_w.size(2))
                outside_w =  outside_w.reshape(-1, outside_w.size(2))

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


class RCNNModule(nn.Module):
    def __init__(self, cfg):
        super(RCNNModule, self).__init__()

        self.inchannels = cfg.inchannels
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
        self.stride = cfg.stride
        self.with_avg_pool = cfg.with_avg_pool
        self.n_classes = len(cfg.class_list[0]) + 1
        self.class_agnostic = cfg.class_agnostic
        self.ohem_rcnn = cfg.ohem_rcnn
        self.pooling_size = cfg.pooling_size
        self.roi_feat_area = self.pooling_size * self.pooling_size

        self.pre_linear = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.inchannels * self.roi_feat_area, rcnn_first)),
            ('fc1_r', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(rcnn_first, rcnn_last_din, bias=True)),
            ('fc2_r', nn.ReLU(inplace=True))
        ]))
        self.cls_score = nn.Linear(rcnn_last_din, self.n_classes)
        out_dim_reg = 4 if self.class_agnostic else 4 * self.n_classes
        self.bbox_pred = nn.Linear(rcnn_last_din, out_dim_reg)
        self.__init_weight__()

        self.proposal_target = ProposalTargetLayer(mean, std, inside_weight, batch_size, fraction, self.n_classes,
                                                   fg_thresh, bg_thresh_hi, bg_thresh_lo)
        self.roi_layers = self.build_roi_layers(self.stride, mode)

    def build_roi_layers(self, strides, mode):
        roi_layers = []
        for stride in strides:
            if mode == 'align':
                roi_pool = ROIAlignLayer(
                    (self.pooling_size, self.pooling_size), 1.0 / stride, 2)
            elif mode == 'pool':
                self.roi_pool = ROIPoolingLayer(
                    (self.pooling_size, self.pooling_size), 1.0 / stride)
            else:
                print('none mode')
                exit(-1)
            roi_layers.append(roi_pool)
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_extract(self, feats, rois):

        extact_feats = feats[:len(self.stride)]
        # out_size = self.roi_layers[0].out_size
        num_levels = len(extact_feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.zeros((rois.size(0), self.inchannels, self.pooling_size, self.pooling_size), device=feats[0].device).float()
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](extact_feats[i], rois_)
                roi_feats[inds] = roi_feats_t

        return roi_feats

    def __init_weight__(self):
        kaiming_init(self.pre_linear.fc1)
        kaiming_init(self.pre_linear.fc2)
        kaiming_init(self.cls_score)
        kaiming_init(self.bbox_pred)

    def forward(self, feats, rois, im_info, gt_boxes):
        batch_size = im_info.size(0)
        with torch.no_grad():
            if self.training:
                roi_data = self.proposal_target(rois, gt_boxes)
                rois, labels, bbox_target, inside_w, outside_w = roi_data

                rois = rois.reshape(-1, 5)
                labels = labels.reshape(-1).long()
                bbox_target = bbox_target.reshape(-1, 4)
                inside_w = inside_w.reshape(-1, 4)
                outside_w =  outside_w.reshape(-1, 4)


        roi_feats = self.roi_extract(feats, rois)
        roi_feats = roi_feats.reshape(roi_feats.size(0), -1)
        pre_linear = self.pre_linear(roi_feats)
        bbox_pred = self.bbox_pred(pre_linear)
        cls_score = self.cls_score(pre_linear)
        cls_prob = F.softmax(cls_score, 1)

        if self.training:
            loss_cls = F.cross_entropy(cls_score, labels)
            bbox_pred = bbox_pred.reshape(-1, 4)
            loss_bbox = _smooth_l1_loss(bbox_pred, bbox_target, inside_w, outside_w)
            return rois, cls_prob, bbox_pred, loss_cls, loss_bbox
        else:
            return rois, cls_prob, bbox_pred





def get_model(config):
    stride = config.stride
    if isinstance(stride, list):
        return RCNNModule(config)
    else:
        return RCNNSingleModule(config)