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
#from models.config import cfg
from .bbox_transform import bbox_transform_inv, clip_boxes
#from lib.roi_layers.nms import nms
import torch.nn.functional as F
from torchvision.ops import nms

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

    def bbox_transform_inv(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def clip_boxes(self, boxes, im_shape):

        boxes[:, 0::4].clamp_(0, im_shape[1] - 1)
        boxes[:, 1::4].clamp_(0, im_shape[0] - 1)
        boxes[:, 2::4].clamp_(0, im_shape[1] - 1)
        boxes[:, 3::4].clamp_(0, im_shape[0] - 1)

        return boxes

    def _filter_boxes(self, boxes, min_size, im_info):
        """Remove all boxes with any side smaller than min_size."""
        x_min_size = min_size * im_info[3]
        y_min_size = min_size * im_info[2]
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = ((ws >= x_min_size) & (hs >= y_min_size))
        return keep

    def forward(self, scores, bbox_deltas, anchors, num_anchors, im_info, pre_nms, post_nms, nms_thresh):

        batch_size = bbox_deltas.size(0)

        batch_size, feat_height, feat_width = scores.size(0), scores.size(2), scores.size(3)

        # anchors = anchors.reshape(1, -1, 4).expand(batch_size, -1, 4)
        anchors = anchors.reshape(-1, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # Same story for the scores:
        # scores = F.softmax(scores.reshape(batch_size, 2, -1, feat_width), dim=1)
        # scores =  scores.reshape(batch_size, num_anchors *2 , -1, feat_width)[:, num_anchors:, :, :]
        # scores = scores.permute(0, 2, 3, 1).reshape(batch_size, -1)

        scores = F.softmax(scores.permute(0, 2, 3, 1).reshape(batch_size, -1, 2), dim=2)[:, :, 1]

        output = torch.zeros((batch_size, post_nms, 5)).to(scores.device)

        for i in range(batch_size):

            # Convert anchors into proposals via bbox transformations
            proposal = self.bbox_transform_inv(anchors, bbox_deltas[i])

            # 2. clip predicted boxes to image
            proposal = self.clip_boxes(proposal, im_info[i])

            score = scores[i]

            keep = self._filter_boxes(proposal, self.min_size, im_info[i])
            # #print(proposals_single.shape, keep.shape)
            proposal = proposal[keep]
            score = score[keep]
            _, order = torch.sort(score, 0, True)

            if pre_nms < score.size(0):
                order = order[:pre_nms]

            proposal = proposal[order]
            score = score[order].reshape(-1)

            keep_idx = nms(proposal, score, nms_thresh).long()

            if post_nms > 0:
                keep_idx = keep_idx[:post_nms]
            proposal = proposal[keep_idx]
            score = score[keep_idx]

            num_proposal = proposal.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposal

        return output

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
