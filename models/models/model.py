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
from models.features import liteHyper
from models.mobilenet_v2 import MobileNet2
from lib.rpn.rpn_regression import rpn_regression
from lib.roi_layers.roi_align_layer import ROIAlignLayer
from lib.roi_layers.roi_pooling_layer import ROIPoolingLayer
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from models.smoothl1loss import smooth_l1_loss
import math

class faster_rcnn(nn.Module):
    '''
    author: syshen
    '''
    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_rcnn_weights(self):
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
        #normal_init(self.Regression.fc6, 0, 0.005, cfg.TRAIN.TRUNCATED)
        #normal_init(self.Regression.fc7, 0, 0.005, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_predict.fc_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_predict.fc_bbox, 0, 0.001, cfg.TRAIN.TRUNCATED)
    def __init__(self, classes, pretrained=False, align=False):
        super(faster_rcnn, self).__init__()        
        self.classes = classes
        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0
        
        self.rpn_regression = rpn_regression(self.rpn_inchannels)
        self.proposallayer = ProposalLayer(cfg.FEAT_STRIDE[0], \
                                           cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
        self.proposaltargetlayer = ProposalTargetLayer(self.classes)
        self.roi_extraction = ROIPoolingLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        if not align:
            self.roi_extraction = ROIAlignLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        self.regressionDim = 512
        self.ROIDim = 256

        self.Regression = nn.Sequential(OrderedDict([
            ('fc6',nn.Linear(self.ROIDim * cfg.POOLING_SIZE * cfg.POOLING_SIZE, self.regressionDim)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc7', nn.Linear(self.regressionDim, self.regressionDim, bias=True)),
            ('fc7_relu', nn.ReLU(inplace=True))]))

        self.cls_predict = nn.Sequential(OrderedDict([
            ('fc_cls',nn.Linear(self.regressionDim, self.classes))]))

        self.bbox_predict = nn.Sequential(OrderedDict([
            ('fc_bbox',nn.Linear(self.regressionDim, self.classes * 4))]))

        self.out_sigmoid = nn.Sigmoid()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def put_values_weights(self):
        self._init_weights()
        self._init_rcnn_weights()

    def forward(self, im_data, im_info, gt_boxes):
        
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        features = self.baskstone(im_data)
        self.labels = None
        self.bbox_targets = None
        self.bbox_inside_weights = None
        self.bbox_outside_weights = None
        self.rois = None
        self.rpn_loss_cls = 0
        self.rpn_loss_bbox = 0
        self.loss_cls  = 0
        self.loss_bbox = 0 
        self.cls_prob = 0
        self.bbox_pred = 0
  
        cfg_key = 'TRAIN' if self.training else 'TEST'
 
        base_feat, rpn_cls_score, rpn_bbox_pred, self.rpn_loss_cls, self.rpn_loss_bbox = \
            self.rpn_regression(features, im_info, gt_boxes)

        self.rois = self.proposallayer((rpn_cls_score.data, rpn_bbox_pred.data, im_info, cfg_key))

        if self.training:
           
            self.rois, self.labels, self.bbox_targets, \
                self.bbox_inside_weights, self.bbox_outside_weights = \
                self.proposaltargetlayer(self.rois, gt_boxes)
     
            self.labels = Variable(self.labels.view(-1).long())
            self.bbox_targets = Variable(self.bbox_targets.view(-1, self.bbox_targets.size(2)))
            self.bbox_inside_weights = Variable(self.bbox_inside_weights.\
                                           view(-1, self.bbox_inside_weights.size(2)))
            self.bbox_outside_weights = Variable(self.bbox_outside_weights.\
                                            view(-1, self.bbox_outside_weights.size(2)))
            

        self.rois = Variable(self.rois)
        # do roi pooling based on predicted rois

        self.roi_feat = self.roi_extraction(base_feat, self.rois.view(-1, 5))

        roi_feat = self.roi_feat.view(self.roi_feat.size(0), -1)  # Reshape into (batchsize, all)      
        inner_product = self.Regression(roi_feat)
        self.bbox_pred = self.bbox_predict(inner_product)
        cls_score = self.cls_predict(inner_product)
        #self.cls_prob = F.softmax(cls_score, 1)
        self.cls_prob = self.out_sigmoid(cls_score)                  

        if self.training:
            self.loss_cls = F.cross_entropy(self.cls_prob, self.labels)
            self.loss_bbox = smooth_l1_loss(self.bbox_pred, self.bbox_targets, self.bbox_inside_weights, \
                                       self.bbox_outside_weights)

        self.cls_prob = self.cls_prob.view(batch_size, self.rois.size(1), -1)
        self.bbox_pred = self.bbox_pred.view(batch_size, self.rois.size(1), -1)
        #print(cls_prob.shape, bbox_pred.shape)

        return self.rois, self.cls_prob, self.bbox_pred, self.rpn_loss_cls, self.rpn_loss_bbox, self.loss_cls, self.loss_bbox

