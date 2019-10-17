from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
#from models.config import cfg
from lib.rpn.generate_anchors import generate_anchors
from lib.rpn.bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
#from lib.roi_layers.nms import nms
import torch.nn.functional as F
from torchvision.ops import nms
import pdb

DEBUG = False


class ProposalLayer(nn.Module):

    def __init__(self, feat_stride, nms_thresh, min_size):
        super(ProposalLayer, self).__init__()

        # self.cfg = cfg
        # self.pre_nms = pre_nms
        # self.post_nms = post_nms
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self._feat_stride = feat_stride

    def forward(self, scores, bbox_deltas, anchors, num_anchors, im_info, pre_nms, post_nms, nms_thresh):

        batch_size = bbox_deltas.size(0)

        batch_size, _, _, feat_width = scores.size()

        anchors = anchors.reshape(1, -1, 4).expand(batch_size, -1, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # Same story for the scores:
        scores = F.softmax(scores.reshape(batch_size, 2, -1, feat_width), dim=1)
        scores =  scores.reshape(batch_size, num_anchors *2 , -1, feat_width)[:, num_anchors:, :, :]
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)

        # scores_keep = scores
        # proposals_keep = proposals
        # _, order = torch.sort(scores, 1, True)
        output = torch.zeros((batch_size, post_nms, 5)).to(scores.device)

        for i in range(batch_size):

            proposals_single = proposals[i]
            scores_single = scores[i]

            keep = self._filter_boxes(proposals_single, self.min_size, im_info[i])
            # #print(proposals_single.shape, keep.shape)
            proposals_single = proposals_single[keep]
            scores_single = scores_single[keep]
            _, order_single = torch.sort(scores_single, 0, True)

            if pre_nms < scores_single.size(0):
                order_single = order_single[:pre_nms]

            proposals_single = proposals_single[order_single]
            scores_single = scores_single[order_single].reshape(-1)

            keep_idx_i = nms(proposals_single, scores_single, nms_thresh).long()

            if post_nms > 0:
                keep_idx_i = keep_idx_i[:post_nms]
            proposals_single = proposals_single[keep_idx_i]
            scores_single = scores_single[keep_idx_i]

            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

        return output, proposals

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size, im_info):
        """Remove all boxes with any side smaller than min_size."""
        x_min_size = min_size * im_info[3]
        y_min_size = min_size * im_info[2]
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = ((ws >= x_min_size) & (hs >= y_min_size))
        return keep
