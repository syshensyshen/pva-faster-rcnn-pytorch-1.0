import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
#from models.config import cfg
from lib.rpn.rpn_regression import _RPN
from collections import OrderedDict
from torchvision.ops import RoIPool as ROIPoolingLayer
from torchvision.ops import RoIAlign as ROIAlignLayer
#from lib.roi_layers.roi_align_layer import ROIAlignLayer
#from lib.roi_layers.roi_pooling_layer import ROIPoolingLayer
from lib.rpn.proposal_target_layer import _ProposalTargetLayer
import time
import pdb
from models.smoothl1loss import _smooth_l1_loss


class _hyper_rcnn(nn.Module):
    """ faster RCNN """

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def __init__(self, cfg, classes, class_agnostic):
        super(_hyper_rcnn, self).__init__()

        self.cfg = cfg

        #self.classes = classes
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.rcnn_din = 4096
        self.rpn_din = 512

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, self.rpn_din)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.downSample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.RCNN_roi_pool = ROIPoolingLayer(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlignLayer(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.downBeat = nn.Conv2d(3584, self.rpn_din, kernel_size=1)
        self.RCNN_top = nn.Sequential(OrderedDict([
            ('fc6_new', nn.Linear(self.dout_base_model *
                                  cfg.POOLING_SIZE * cfg.POOLING_SIZE, self.rcnn_din)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc7_new', nn.Linear(self.rcnn_din, self.rcnn_din, bias=True)),
            ('fc7_relu', nn.ReLU(inplace=True))]))

    def forward(self, im_data, im_info, gt_boxes):
        cfg = self.cfg

        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        #num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)  # 1/8
        base_feat2 = self.RCNN_base2(base_feat1)  # 1/16
        base_feat3 = self.RCNN_base3(base_feat2)  # 1/32
        downSample = self.downSample(base_feat1)
        upSample = F.interpolate(base_feat3, scale_factor=2, mode='nearest')

        base_feat = torch.cat((downSample, base_feat2, upSample), 1)
        base_feat = self.downBeat(base_feat)

        # print(base_feat.shape)
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # print(self.training)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(
                rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # print(rois.shape)
        # for index in range(0, 300):
        #    if cls_prob[:,index,0] < 0.5:
        #        print(cls_prob[:,index,:], rois[:,index,:])
        # print(bbox_pred)
        if self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        else:
            return rois, cls_prob, bbox_pred

    def _init_weights(self, cfg):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score,
                    0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred,
                    0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self, cfg):
        self._init_modules(cfg)
        self._init_weights(cfg)
