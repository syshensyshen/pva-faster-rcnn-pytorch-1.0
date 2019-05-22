import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from models.config import cfg
from lib.rpn.rpn_regression import _RPN
from collections import OrderedDict
from lib.roi_layers.roi_align_layer import ROIAlignLayer
from lib.roi_layers.roi_pooling_layer import ROIPoolingLayer
from lib.rpn.proposal_target_layer import _ProposalTargetLayer
import time
import pdb
from models.smoothl1loss import _smooth_l1_loss

class pva_faster_rcnn(nn.Module):
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval() 
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(pva_faster_rcnn, self).__init__()
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.rcnn_din = cfg.MODEL.RCNN_CIN
        self.rcnn_last_din = cfg.MODEL.RCNN_LAST
        self.rpn_din = cfg.MODEL.RPN_CIN

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, self.rpn_din)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        if cfg.POOLING_MODE == 'align':
            self.RCNN_roi_pool = ROIAlignLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / cfg.FEAT_STRIDE[0], 2)
        elif cfg.POOLING_MODE == 'pool':
            self.RCNN_roi_pool = ROIPoolingLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / cfg.FEAT_STRIDE[0])

        self.RCNN_top = nn.Sequential(OrderedDict([
            ('fc6_new',nn.Linear(self.dout_base_model * cfg.POOLING_SIZE * cfg.POOLING_SIZE, self.rcnn_din)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc7_new', nn.Linear(self.rcnn_din, self.rcnn_last_din, bias=True)),
            ('fc7_relu', nn.ReLU(inplace=True))
            ]))

    def forward(self, im_data, im_info, gt_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        #num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #print(base_feat.shape)
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
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
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        #print(self.training)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            #return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox
        else:
            return rois, cls_prob, bbox_pred

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

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_top.fc6_new, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_top.fc7_new, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
