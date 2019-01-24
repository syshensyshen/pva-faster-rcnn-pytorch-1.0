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
from lib.roi_layers.roi_align_layer import ROIAlignLayer
from lib.roi_layers.roi_pooling_layer import ROIPoolingLayer
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
        normal_init(self.Regression.fc6, 0, 0.005, cfg.TRAIN.TRUNCATED)
        normal_init(self.Regression.fc7, 0, 0.005, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_prob.fc_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred.fc_bbox, 0, 0.001, cfg.TRAIN.TRUNCATED)
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
        self.roi_extraction = ROIPoolingLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        if not align:
            self.roi_extraction = ROIAlignLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        if pretrain:
            model = torch.load(cfg.TRAIN.PRETRAINEDMODEL)
            self.Backstone = copy.deepcopy(model.state_dict())

        self.regressionDim = 512
        self.ROIDim = 256
        #self.roi_pooling_size = cfg.POOLING_SIZE

        self.Regression = nn.Sequential(OrderedDict([
            ('fc6',nn.Linear(self.ROIDim * cfg.POOLING_SIZE * cfg.POOLING_SIZE, self.regressionDim)),
            ('fc6_relu', nn.LeakyReLU(inplace=True)),
            ('fc7', nn.Linear(self.regressionDim, self.regressionDim, bias=True)),
            ('fc7_relu', nn.LeakyReLU(inplace=True))]))

        self.cls_prob = nn.Sequential(OrderedDict([
            ('fc_cls',nn.Linear(self.regressionDim, self.classes))]))

        self.bbox_pred = nn.Sequential(OrderedDict([
            ('fc_bbox',nn.Linear(self.regressionDim, self.classes * 4))]))
        self._init_weights()

    def forward(self, im_data, im_info, gt_boxes):
        #print(im_data.shape)
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        #num_boxes = num_boxes.data
        features = self.Backstone(im_data)
        #print(features.shape)
        self.labels = None
        self.bbox_targets = None
        self.bbox_inside_weights = None
        self.bbox_outside_weights = None
        self.rois = None
        self.rpn_loss_cls = 0
        self.rpn_loss_bbox = 0
        cfg_key = 'TRAIN' if self.training else 'TEST'
        base_feat, rpn_cls_score, rpn_bbox_pred, rpn_loss_cls, rpn_loss_bbox = \
            self.rpn_regression(features, im_info, gt_boxes)
        #print(base_feat.shape)
        self.rois = self.proposallayer((rpn_cls_score.data, rpn_bbox_pred.data, im_info, cfg_key))
        if self.training:
            #print(self.rois.shape, gt_boxes.shape)           
            self.rois, self.labels, self.bbox_targets, \
                self.bbox_inside_weights, self.bbox_outside_weights = \
                self.proposaltargetlayer(self.rois, gt_boxes)
            #print(self.bbox_targets.shape)
            self.labels = Variable(self.labels.view(-1).long())
            self.bbox_targets = Variable(self.bbox_targets.view(-1, self.bbox_targets.size(2)))
            self.bbox_inside_weights = Variable(self.bbox_inside_weights.\
                                           view(-1, self.bbox_inside_weights.size(2)))
            self.bbox_outside_weights = Variable(self.bbox_outside_weights.\
                                            view(-1, self.bbox_outside_weights.size(2)))
            

        self.rois = Variable(self.rois)
         # do roi pooling based on predicted rois
        roi_feat = self.roi_extraction(base_feat, self.rois.view(-1, 5))
        roi_feat = roi_feat.view(roi_feat.size(0), -1)  # Reshape into (batchsize, all)      
        inner_product = self.Regression(roi_feat)
        bbox_pred = self.bbox_pred(inner_product)
        cls_score = self.cls_prob(inner_product)
        cls_prob = F.softmax(cls_score, 1)
        #cls_prob = torch.sigmoid(cls_score)

        #if self.training:
        #    bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #    bbox_pred_select = torch.gather(bbox_pred_view, 1, self.labels.view(self.labels.size(0), 1, 1).expand(self.labels.size(0), 1, 4))
        #    bbox_pred = bbox_pred_select.squeeze(1)

        #print(bbox_pred.shape, self.bbox_targets.shape, cls_prob.shape)

        loss_cls  = 0
        loss_bbox = 0                

        if self.training:
            loss_cls = F.cross_entropy(cls_prob, self.labels)
            loss_bbox = smooth_l1_loss(bbox_pred, self.bbox_targets, self.bbox_inside_weights, \
                                       self.bbox_outside_weights)

        cls_prob = cls_prob.view(batch_size, self.rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, self.rois.size(1), -1)

        return self.rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox

