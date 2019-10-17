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
import numpy.random as npr

# from models.config import cfg
from lib.rpn.generate_anchors import generate_anchors
from lib.rpn.bbox_transform import clip_boxes, clip_boxes_batch
from lib.rpn.bbox_transform import bbox_transform_batch, bbox_transform, bbox_overlaps_batch, bbox_overlaps

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """

    def __init__(self, batch_size, negative_overlap, positive_overlap, positive_weight=1.0, fraction=0.5, inside_weight=1.0, clobber_positive=False):
        super(AnchorTargetLayer, self).__init__()

        self.clobber_positive = clobber_positive
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap
        self.batch_size = batch_size
        self.positive_weight = positive_weight
        self.fraction = fraction
        self.inside_weight = inside_weight
        self._allowed_border = 0  # default is 0

    def forward(self, rpn_cls_score, all_anchors, num_anchors, im_info, gt_boxes=None):

        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        # feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        total_anchors = all_anchors.size(0)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = []
        bbox_targets = []
        bbox_inside_weights = []
        bbox_outside_weights = []

        for i in range(batch_size):
            vaild_anchors = inds_inside.size(0)
            label = torch.zeros(
                (vaild_anchors), dtype=torch.float32, device=gt_boxes[i].device).fill_(-1)
            bbox_inside_weight = torch.zeros(
                (vaild_anchors), dtype=torch.float32, device=gt_boxes[i].device)
            bbox_outside_weight = torch.zeros(
                (vaild_anchors), dtype=torch.float32, device=gt_boxes[i].device)
            overlap = bbox_overlaps(anchors, gt_boxes[i][:, 0:4])
            max_overlaps, argmax_overlaps = torch.max(overlap, 1)
            gt_max_overlaps, _ = torch.max(overlap, 0)

            if not self.clobber_positive:
                if (max_overlaps < self.positive_overlap).size(0) <= label.size(0):
                    label[max_overlaps < self.negative_overlap] = 0
            gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
            keep = torch.sum(overlap.eq(gt_max_overlaps.view(1, -1).expand_as(overlap)), 1)

            if torch.sum(keep) > 0:
                label[keep > 0] = 1

            label[max_overlaps >= self.positive_overlap] = 1

            if self.clobber_positive:
                label[max_overlaps < self.negative_overlap] = 0

            num_fg = int(self.fraction * self.batch_size)

            sum_fg = torch.sum((label == 1).int(), 0)
            sum_bg = torch.sum((label == 0).int(), 0)
            # print("=> sum_fg: ", sum_fg, " sum_bg: ", sum_bg)

            if sum_fg > num_fg:
                fg_inds = torch.nonzero(label == 1).reshape(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0)))
                rand_num = rand_num.to(gt_boxes[i].device).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                label[disable_inds] = -1
                sum_fg = num_fg

            num_bg = self.batch_size - sum_fg

            if sum_bg > num_bg:
                bg_inds = torch.nonzero(label == 0).view(-1)

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                label[disable_inds] = -1
                sum_bg = num_bg

            # print("=> sum_fg: ", sum_fg, " sum_bg: ", sum_bg)

            bbox_target = _compute_targets(
                anchors, gt_boxes[i][argmax_overlaps.view(-1), :].reshape(-1, 5))

            # use a single value instead of 4 values for easy index.
            bbox_inside_weight[label ==1] = self.inside_weight

            # assert ((self.positive_weight > 0) & (self.positive_weight < 1))

            num_examples = torch.sum(label >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
            bbox_outside_weight[label == 1] = positive_weights
            bbox_outside_weight[label == 0] = negative_weights
            labels.append(label)
            bbox_targets.append(bbox_target)
            bbox_inside_weights.append(bbox_inside_weight)
            bbox_outside_weights.append(bbox_outside_weight)

        labels = torch.cat(labels).reshape(batch_size, inds_inside.size(0))
        bbox_targets = torch.cat(bbox_targets).reshape(batch_size, inds_inside.size(0), -1)
        bbox_inside_weights = torch.cat(bbox_inside_weights).reshape(batch_size, inds_inside.size(0), -1)
        bbox_outside_weights = torch.cat(bbox_outside_weights).reshape(batch_size, inds_inside.size(0), -1)

        labels = _unmap(labels, total_anchors,inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors,inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        # outputs = []

        labels = labels.reshape(batch_size, height, width, num_anchors).permute(0, 3, 1, 2)
        labels = labels.reshape(batch_size, 1, num_anchors * height, width)
        # outputs.append(labels)

        bbox_targets = bbox_targets.reshape(batch_size, height, width, num_anchors*4).permute(0, 3, 1, 2)
        # outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.reshape(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_inside_weights = bbox_inside_weights.reshape(batch_size, height, width, 4*num_anchors).permute(0, 3, 1, 2)

        # outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.reshape(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.reshape(batch_size, height, width, 4*num_anchors).permute(0, 3, 1, 2)
        # outputs.append(bbox_outside_weights)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)
                           ).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):

    return bbox_transform(ex_rois, gt_rois[:, 0:4])