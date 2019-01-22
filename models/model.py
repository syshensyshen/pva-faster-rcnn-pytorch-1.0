# -*- coding: utf-8 -*-
'''
# @Author  : syshen 
# @date    : 2018/12/20 18:19
# @File    : pva network as backstone in faster-rcnn
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import copy
import numpy as np
from models.config import cfg
from models.features import pvaHyper
from lib.rpn.rpn_regression import rpn_regression
from lib.rpn.roi_align_layer import ROIAlignLayer
from lib.rpn.roi_pooling_layer import ROIPoolingLayer
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from models.smoothl1loss import smooth_l1_loss

class network(nn.Module):
    '''
    author: syshen
    '''
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.cls_inner, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
    def __init__(self, classes, pretrain=False, align=False):
        super(network, self).__init__()        
        self.classes = classes
        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0
        self.Backstone = pvaHyper() # pva in_channles: 768 lite inchannels: 544
        self.features = self.Backstone
        self.rpn_regression = rpn_regression(768)
        self.proposallayer = ProposalLayer(cfg.FEAT_STRIDE[0], \
                                           cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
        self.proposaltargetlayer = ProposalTargetLayer(self.classes)
        self.roi_extraction = ROIAlignLayer()
        if not align:
            self.roi_extraction = ROIPoolingLayer()
        if pretrain:
            model = torch.load(cfg.TRAIN.PRETRAINEDMODEL)
            self.Backstone = copy.deepcopy(model.state_dict())

        self.regressionDim = 512
        self.roi_pooling_size = 6

        self.Regression = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim * self.roi_size * self.roi_size, self.regressionDim)),
            ('fc_relu', nn.LeakyReLU(inplace=True)),
            ('fc2', nn.Linear(self.regressionDim, self.regressionDim, bias=True)),
            ('fc2_relu', nn.LeakyReLU(inplace=True))]))

        self.cls_inner = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes))]))

        self.bbox_pred = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes * 4))]))

    def forward(self, base_feat, im_data, im_info, gt_boxes, num_boxes, training=False):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        features = self.Backstone(im_data)
        self.labels = None
        self.bbox_targets = None
        self.bbox_inside_weights = None
        self.bbox_outside_weights = None
        self.rois = None
        self.rpn_loss_cls = 0
        self.rpn_loss_bbox = 0
        base_feat, rpn_cls_score, rpn_bbox_pred, rpn_loss_cls, rpn_loss_bbox = \
            self.rpn_regression.forward(features, im_info, gt_boxes, num_boxes)
        self.rois = self.proposallayer(rpn_cls_score.data, \
            rpn_bbox_pred.data, im_info)
        if training:           
            self.labels, self.rois, self.bbox_targets, self.bbox_inside_weights = \
                self.proposaltargetlayer(self.rois, gt_boxes, num_boxes)
            self.bbox_outside_weights = np.array(self.bbox_inside_weights > 0).astype(np.float32)
            self.rois_label = Variable(self.labels.view(-1).long())
            self.bbox_targets = Variable(self.bbox_targets.view(-1, self.bbox_targets.size(2)))
            self.bbox_inside_weights = Variable(self.bbox_inside_weights.\
                                           view(-1, self.bbox_inside_weights.size(2)))
            self.bbox_outside_weights = Variable(self.bbox_outside_weights.\
                                            view(-1, self.bbox_outside_weights.size(2)))
            

        self.rois = Variable(self.rois)
         # do roi pooling based on predicted rois
        roi_feat = self.roi_extraction(base_feat, self.rois.view(-1, 5), \
                                          self.roi_pooling_size, cfg.TRAIN.FEAT_STRIDE)
        inner_product = self.Regression(roi_feat)
        bbox_pred = self.bbox_pred(inner_product)
        cls_inner = self.cls_inner(inner_product)
        cls_prob = F.softmax(cls_inner, 1)

        loss_cls  = 0
        loss_bbox = 0

        if training:
            loss_cls = nn.CrossEntropyLoss()
            loss_bbox = smooth_l1_loss(bbox_pred, self.bbox_targets, self.bbox_inside_weights, \
                                       self.bbox_outside_weights)

        cls_prob = cls_prob.view(batch_size, self.rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, self.rois.size(1), -1)

        return self.rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox

