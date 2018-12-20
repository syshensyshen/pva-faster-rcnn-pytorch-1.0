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
import numpy as np
from models.pvanet import *
from rpn import _RPN
from roi_align import *
from roi_pool import *
from model.utils.config import cfg

class pvaHyper(nn.Module):
    '''
    '''
    def __init__(self):
        super(PVANetFeat, self).__init__()
        
    def forward(self, input):
        x0 = self.conv1(input)
        x1 = self.conv2(x0)         # 1/4 feature
        x2 = self.conv3(x1)         # 1/8
        x3 = self.conv4(x2)         # 1/16
        x4 = self.conv5(x3)         # 1/32
        downsample = F.avg_pool2d(x2, , kernel_size=3, stride=2, padding=0).view(x2.size(0), -1)
        upsample = F.interpolate(x4, scale_factor=2, mode="nearest")
        features = torch.cat((downsample, x3, upsample), 1)
        return features
    

class network(nn.Module):
    '''
    author: syshen
    '''
    def __init__(self, classes, pretrain=False, align=False):
        super(network, self).__init__()        
        self.classes = classes
        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0
        self.Backstone = pvaHyper()
        self.features = self.Backstone
        self.rpn = _rpn()
        #self.proposal = _ProposalLayer()
        self.proposaltarget = _ProposalTargetLayer()
        self.roi_extraction = _ROIAlign() if align else: _ROIPool()
        if pretrain:
            model = torch.load(cfg.TRAIN.pretainmodel)
            self.Backstone = copy.deepcopy(model.state_dict())

        self.regressionDim = 512
        self.roi_size = 6

        self.Regression = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim * self.roi_size * self.roi_size, self.regressionDim)),
            ('fc_relu', nn.LeakyReLU(inplace=True)),
            ('fc2', nn.Linear(self.regressionDim, self.regressionDim, bias=True)),
            ('fc2_relu', nn.LeakyReLU(inplace=True))]))

        self.cls_inner = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes))]))

        self.bbox_pred = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes * 4))]))

    def forward(self, im_data, im_info, gt_boxes, num_boxes, training=False):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        features = self.Backstone(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(features, im_info, gt_boxes, num_boxes)
        if training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
         # do roi pooling based on predicted rois
        pooled_feat = self.roi_extraction(base_feat, rois.view(-1, 5), self.roi_size, cfg.TRAIN.spatial_scale)
        inner_product = self.Regression(pooled_feat)
        bbox_pred = self.bbox_pred(inner_product)
        cls_inner = self.cls_inner(inner_product)
        cls_prob = F.softmax(cls_prob, 1)

        loss_cls  = 0
        loss_bbox = 0

        if training:
            loss_cls = nn.CrossEntropyLoss()
            loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, rois_label     
