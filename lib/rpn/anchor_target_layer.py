# from __future__ import absolute_import
# # --------------------------------------------------------
# # Faster R-CNN
# # Copyright (c) 2015 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ross Girshick and Sean Bell
# # --------------------------------------------------------
# # --------------------------------------------------------
# # Reorganized and modified by Jianwei Yang and Jiasen Lu
# # --------------------------------------------------------

# import torch
# import torch.nn as nn
# import numpy as np
# import numpy.random as npr

# #from models.config import cfg
# from lib.rpn.generate_anchors import generate_anchors
# from lib.rpn.bbox_transform import clip_boxes, clip_boxes_batch
# from lib.rpn.bbox_transform import bbox_transform_batch, bbox_overlaps_batch

# import pdb

# DEBUG = False

# try:
#     long        # Python 2
# except NameError:
#     long = int  # Python 3


# class AnchorTargetLayer(nn.Module):
#     """
#         Assign anchors to ground-truth targets. Produces anchor classification
#         labels and bounding-box regression targets.
#     """

#     def __init__(self, cfg, feat_stride, scales, ratios):
#         super(AnchorTargetLayer, self).__init__()

#         self.cfg = cfg

#         self._feat_stride = feat_stride
#         self._scales = scales
#         anchor_scales = scales
#         self._anchors = torch.from_numpy(generate_anchors(base_size=cfg.FEAT_STRIDE[0], scales=np.array(anchor_scales),
#                                                           ratios=np.array(ratios))).float()
#         self._num_anchors = self._anchors.size(0)

#         # allow boxes to sit over the edge by a small amount
#         self._allowed_border = 0  # default is 0

#     def forward_ssy(self, input):
#         # Algorithm:
#         #
#         # for each (H, W) location i
#         #   generate 9 anchor boxes centered on cell i
#         #   apply predicted bbox deltas at cell i to each of the 9 anchors
#         # filter out-of-image anchors

#         cfg = self.cfg

#         RPN_cls_score = input[0]
#         gt_boxes = input[1]
#         im_info = input[2]
#         #num_boxes = input[3]

#         # map of shape (..., H, W)
#         height, width = RPN_cls_score.size(2), RPN_cls_score.size(3)

#         batch_size = gt_boxes.size(0)

#         feat_height, feat_width = RPN_cls_score.size(
#             2), RPN_cls_score.size(3)
#         shift_x = np.arange(0, feat_width) * self._feat_stride
#         shift_y = np.arange(0, feat_height) * self._feat_stride
#         shift_x, shift_y = np.meshgrid(shift_x, shift_y)
#         shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
#                                              shift_x.ravel(), shift_y.ravel())).transpose())
#         shifts = shifts.contiguous().type_as(RPN_cls_score).float()

#         A = self._num_anchors
#         K = shifts.size(0)

#         # move to specific gpu.
#         self._anchors = self._anchors.type_as(gt_boxes)
#         all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
#         all_anchors = all_anchors.view(K * A, 4)

#         total_anchors = int(K * A)

#         keep = ((all_anchors[:, 0] >= -self._allowed_border) &
#                 (all_anchors[:, 1] >= -self._allowed_border) &
#                 (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
#                 (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

#         inds_inside = torch.nonzero(keep).view(-1)

#         # keep only inside anchors
#         anchors = all_anchors[inds_inside, :]

#         # label: 1 is positive, 0 is negative, -1 is dont care
#         labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
#         bbox_inside_weights = gt_boxes.new(
#             batch_size, inds_inside.size(0)).zero_()
#         bbox_outside_weights = gt_boxes.new(
#             batch_size, inds_inside.size(0)).zero_()

#         overlaps = bbox_overlaps_batch(anchors, gt_boxes)

#         max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
#         gt_max_overlaps, _ = torch.max(overlaps, 1)

#         if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
#             if (max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP).size(1) <= labels.size(1):
#                 labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

#         gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
#         keep = torch.sum(overlaps.eq(gt_max_overlaps.view(
#             batch_size, 1, -1).expand_as(overlaps)), 2)

#         if torch.sum(keep) > 0:
#             labels[keep > 0] = 1

#         # fg label: above threshold IOU
#         labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

#         if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
#             labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

#         num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

#         sum_fg = torch.sum((labels == 1).int(), 1)
#         sum_bg = torch.sum((labels == 0).int(), 1)

#         for i in range(batch_size):
#             # subsample positive labels if we have too many
#             if sum_fg[i] > num_fg:
#                 fg_inds = torch.nonzero(labels[i] == 1).view(-1)
#                 # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
#                 # See https://github.com/pytorch/pytorch/issues/1868 for more details.
#                 # use numpy instead.
#                 #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
#                 rand_num = torch.from_numpy(np.random.permutation(
#                     fg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
#                 labels[i][disable_inds] = -1

#             num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
#             # print(num_bg)
#             # num_bg = 6.0 * torch.sum((labels == 1).int(), 1)[i] #cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]
#             # num_bg = min (max(16, num_bg), cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i])

#             # subsample negative labels if we have too many
#             if sum_bg[i] > num_bg:
#                 bg_inds = torch.nonzero(labels[i] == 0).view(-1)
#                 #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

#                 rand_num = torch.from_numpy(np.random.permutation(
#                     bg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
#                 labels[i][disable_inds] = -1

#         offset = torch.arange(0, batch_size)*gt_boxes.size(1)

#         argmax_overlaps = argmax_overlaps + \
#             offset.view(batch_size, 1).type_as(argmax_overlaps)
#         bbox_targets = _compute_targets_batch(
#             anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

#         # use a single value instead of 4 values for easy index.
#         bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

#         if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
#             num_examples = torch.sum(labels[i] >= 0)
#             positive_weights = 1.0 / num_examples.item()
#             negative_weights = 1.0 / num_examples.item()
#         else:
#             assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
#                     (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

#         bbox_outside_weights[labels == 1] = positive_weights
#         bbox_outside_weights[labels == 0] = negative_weights

#         labels = _unmap(labels, total_anchors,
#                         inds_inside, batch_size, fill=-1)
#         bbox_targets = _unmap(bbox_targets, total_anchors,
#                               inds_inside, batch_size, fill=0)
#         bbox_inside_weights = _unmap(
#             bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
#         bbox_outside_weights = _unmap(
#             bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

#         outputs = []

#         labels = labels.view(batch_size, height, width,
#                              A).permute(0, 3, 1, 2).contiguous()
#         labels = labels.view(batch_size, 1, A * height, width)
#         outputs.append(labels)

#         bbox_targets = bbox_targets.view(
#             batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()
#         outputs.append(bbox_targets)

#         anchors_count = bbox_inside_weights.size(1)
#         bbox_inside_weights = bbox_inside_weights.view(
#             batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)

#         bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
#             .permute(0, 3, 1, 2).contiguous()

#         outputs.append(bbox_inside_weights)

#         bbox_outside_weights = bbox_outside_weights.view(
#             batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
#         bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
#             .permute(0, 3, 1, 2).contiguous()
#         outputs.append(bbox_outside_weights)

#         return outputs

#     def forward_zhangcc(self, input):
#         # Algorithm:
#         #
#         # for each (H, W) location i
#         #   generate 9 anchor boxes centered on cell i
#         #   apply predicted bbox deltas at cell i to each of the 9 anchors
#         # filter out-of-image anchors

#         cfg = self.cfg

#         RPN_cls_score = input[0]
#         gt_boxes = input[1]
#         im_info = input[2]
#         #num_boxes = input[3]

#         # map of shape (..., H, W)
#         height, width = RPN_cls_score.size(2), RPN_cls_score.size(3)

#         batch_size = gt_boxes.size(0)

#         feat_height, feat_width = RPN_cls_score.size(
#             2), RPN_cls_score.size(3)
#         shift_x = np.arange(0, feat_width) * self._feat_stride
#         shift_y = np.arange(0, feat_height) * self._feat_stride
#         shift_x, shift_y = np.meshgrid(shift_x, shift_y)
#         shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
#                                              shift_x.ravel(), shift_y.ravel())).transpose())
#         shifts = shifts.contiguous().type_as(RPN_cls_score).float()

#         A = self._num_anchors
#         K = shifts.size(0)

#         # move to specific gpu.
#         self._anchors = self._anchors.type_as(gt_boxes)
#         all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
#         all_anchors = all_anchors.view(K * A, 4)

#         total_anchors = int(K * A)

#         keep = ((all_anchors[:, 0] >= -self._allowed_border) &
#                 (all_anchors[:, 1] >= -self._allowed_border) &
#                 (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
#                 (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

#         inds_inside = torch.nonzero(keep).view(-1)

#         # keep only inside anchors
#         anchors = all_anchors[inds_inside, :]

#         # label: 1 is positive, 0 is negative, -1 is dont care
#         labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
#         bbox_inside_weights = gt_boxes.new(
#             batch_size, inds_inside.size(0)).zero_()
#         bbox_outside_weights = gt_boxes.new(
#             batch_size, inds_inside.size(0)).zero_()

#         outputs = []
#         for i in range(batch_size):
#             overlaps = bbox_overlaps_batch(
#                 anchors, torch.unsqueeze(gt_boxes[i], 0))

#             max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
#             gt_max_overlaps, _ = torch.max(overlaps, 1)

#             if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
#                 if (max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP).size(1) <= labels.size(1):
#                     labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

#             gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
#             keep = torch.sum(overlaps.eq(gt_max_overlaps.view(
#                 1, 1, -1).expand_as(overlaps)), 2)

#             if torch.sum(keep) > 0:
#                 labels[keep > 0] = 1

#             # fg label: above threshold IOU
#             labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

#             if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
#                 labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

#             num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

#             sum_fg = torch.sum((labels == 1).int(), 1)
#             sum_bg = torch.sum((labels == 0).int(), 1)

#             # subsample positive labels if we have too many
#             if sum_fg > num_fg:
#                 fg_inds = torch.nonzero(labels[i] == 1).view(-1)
#                 # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
#                 # See https://github.com/pytorch/pytorch/issues/1868 for more details.
#                 # use numpy instead.
#                 #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
#                 rand_num = torch.from_numpy(np.random.permutation(
#                     fg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
#                 labels[i][disable_inds] = -1

#             num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg
#             # print(num_bg)
#             # num_bg = 6.0 * torch.sum((labels == 1).int(), 1)[i] #cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]
#             # num_bg = min (max(16, num_bg), cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i])

#             # subsample negative labels if we have too many
#             if sum_bg > num_bg:
#                 bg_inds = torch.nonzero(labels[i] == 0).view(-1)
#                 #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

#                 rand_num = torch.from_numpy(np.random.permutation(
#                     bg_inds.size(0))).type_as(gt_boxes).long()
#                 disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
#                 labels[i][disable_inds] = -1

#             offset = torch.arange(0, 1)*gt_boxes.size(1)

#             argmax_overlaps = argmax_overlaps + \
#                 offset.view(1, 1).type_as(argmax_overlaps)
#             bbox_targets = _compute_targets_batch(
#                 anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(1, -1, 5))

#             # use a single value instead of 4 values for easy index.
#             bbox_inside_weights[labels ==
#                                 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

#             if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
#                 num_examples = torch.sum(labels[i] >= 0)
#                 positive_weights = 1.0 / num_examples.item()
#                 negative_weights = 1.0 / num_examples.item()
#             else:
#                 assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
#                         (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

#             bbox_outside_weights[labels == 1] = positive_weights
#             bbox_outside_weights[labels == 0] = negative_weights

#             labels = _unmap(labels, total_anchors,
#                             inds_inside, 1, fill=-1)
#             bbox_targets = _unmap(bbox_targets, total_anchors,
#                                   inds_inside, 1, fill=0)
#             bbox_inside_weights = _unmap(
#                 bbox_inside_weights, total_anchors, inds_inside, 1, fill=0)
#             bbox_outside_weights = _unmap(
#                 bbox_outside_weights, total_anchors, inds_inside, 1, fill=0)

#             labels = labels.view(1, height, width,
#                                  A).permute(0, 3, 1, 2).contiguous()
#             labels = labels.view(1, 1, A * height, width)

#             bbox_targets = bbox_targets.view(
#                 1, height, width, A * 4).permute(0, 3, 1, 2).contiguous()

#             anchors_count = bbox_inside_weights.size(1)
#             bbox_inside_weights = bbox_inside_weights.view(
#                 1, anchors_count, 1).expand(1, anchors_count, 4)

#             bbox_inside_weights = bbox_inside_weights.contiguous().view(1, height, width, 4*A)\
#                 .permute(0, 3, 1, 2).contiguous()

#             bbox_outside_weights = bbox_outside_weights.view(
#                 1, anchors_count, 1).expand(1, anchors_count, 4)
#             bbox_outside_weights = bbox_outside_weights.contiguous().view(1, height, width, 4*A)\
#                 .permute(0, 3, 1, 2).contiguous()

#             outputs.append(labels)
#             outputs.append(bbox_targets)
#             outputs.append(bbox_inside_weights)
#             outputs.append(bbox_outside_weights)

#         return outputs

#     def backward(self, top, propagate_down, bottom):
#         """This layer does not propagate gradients."""
#         pass

#     def reshape(self, bottom, top):
#         """Reshaping happens during the call to forward."""
#         pass


# def _unmap(data, count, inds, batch_size, fill=0):
#     """ Unmap a subset of item (data) back to the original set of items (of
#     size count) """

#     if data.dim() == 2:
#         ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
#         ret[:, inds] = data
#     else:
#         ret = torch.Tensor(batch_size, count, data.size(2)
#                            ).fill_(fill).type_as(data)
#         ret[:, inds, :] = data
#     return ret


# def _compute_targets_batch(ex_rois, gt_rois):
#     """Compute bounding-box regression targets for an image."""

#     return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


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

    def __init__(self, cfg, feat_stride, scales, ratios):
        super(AnchorTargetLayer, self).__init__()

        self.cfg = cfg

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(base_size=cfg.FEAT_STRIDE[0], scales=np.array(anchor_scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        cfg = self.cfg

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        #num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        # move to specific gpu.
        self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)

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

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                if (max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP).size(0) <= label.size(0):
                    label[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
            keep = torch.sum(overlap.eq(
                gt_max_overlaps.view(1, -1).expand_as(overlap)), 1)

            if torch.sum(keep) > 0:
                label[keep > 0] = 1

            label[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                label[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

            sum_fg = torch.sum((label == 1).int(), 0)
            sum_bg = torch.sum((label == 0).int(), 0)
            # print("=> sum_fg: ", sum_fg, " sum_bg: ", sum_bg)

            if sum_fg > num_fg:
                fg_inds = torch.nonzero(label == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(
                    fg_inds.size(0)))
                rand_num = rand_num.to(gt_boxes[i].device).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                label[disable_inds] = -1
                sum_fg = num_fg

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg

            if sum_bg > num_bg:
                bg_inds = torch.nonzero(label == 0).view(-1)

                rand_num = torch.from_numpy(np.random.permutation(
                    bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                label[disable_inds] = -1
                sum_bg = num_bg

            # print("=> sum_fg: ", sum_fg, " sum_bg: ", sum_bg)

            bbox_target = _compute_targets(
                anchors, gt_boxes[i][argmax_overlaps.view(-1), :].view(-1, 5))

            # use a single value instead of 4 values for easy index.
            bbox_inside_weight[label ==
                               1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                num_examples = torch.sum(label >= 0)
                positive_weights = 1.0 / num_examples.item()
                negative_weights = 1.0 / num_examples.item()
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            bbox_outside_weight[label == 1] = positive_weights
            bbox_outside_weight[label == 0] = negative_weights
            labels.append(label)
            bbox_targets.append(bbox_target)
            bbox_inside_weights.append(bbox_inside_weight)
            bbox_outside_weights.append(bbox_outside_weight)

        labels = torch.cat(labels).view(batch_size, inds_inside.size(0))
        bbox_targets = torch.cat(bbox_targets).view(
            batch_size, inds_inside.size(0), -1)
        bbox_inside_weights = torch.cat(bbox_inside_weights).view(
            batch_size, inds_inside.size(0), -1)
        bbox_outside_weights = torch.cat(bbox_outside_weights).view(
            batch_size, inds_inside.size(0), -1)

        labels = _unmap(labels, total_anchors,
                        inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors,
                              inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(
            bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(
            bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width,
                             A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(
            batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(
            batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
            .permute(0, 3, 1, 2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(
            batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
            .permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

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
