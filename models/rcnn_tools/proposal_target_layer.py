from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
#from models.config import cfg
import pdb
from ..rpn_tools.bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class ProposalTargetLayer(nn.Module):
    def __init__(self, mean, std, inside_weight, batch_size, fraction, classes, fg_thresh, bg_thresh_hi, bg_thresh_lo):
        super(ProposalTargetLayer, self).__init__()

        # self.cfg = cfg

        self._num_classes = classes
        self.means = torch.FloatTensor(mean)
        self.stds = torch.FloatTensor(std)
        self.inside_weights = torch.FloatTensor(inside_weight)
        self.batch_size = batch_size
        self.fraction = fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def forward(self, all_rois, gt_boxes):
        with torch.no_grad():
            # cfg = self.cfg
            device = all_rois.device
            self.means = self.means.to(device)
            self.stds = self.stds.to(device)
            self.inside_weights = self.inside_weights.to(device)
            # add gt_boxes in proposal
            new_all_rois = torch.zeros((all_rois.size(0), all_rois.size(1)+gt_boxes.size(1), all_rois.size(2)))
            new_gt_boxes = torch.zeros_like(gt_boxes)
            new_gt_boxes[:, :, 1:5] = gt_boxes[:, :, 0:4]
            for index in range(all_rois.size(0)):
                new_gt_boxes[index, :, 0] = index
            new_all_rois[:, 0:all_rois.size(1), :] = all_rois[:, :, :]
            new_all_rois[:, all_rois.size(1):all_rois.size(1) + new_gt_boxes.size(1), :] = new_gt_boxes

            all_rois = new_all_rois.to(all_rois.device)

            # num_images = 1
            # rois_per_image = int(self.batch_size / num_images)
            # fg_rois_per_image = int(np.round(self.fraction * self.batch_size))

            # fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

            labels, rois, bbox_targets, bbox_inside_weights = \
                self._sample_rois_pytorch(all_rois, gt_boxes, self._num_classes)

            bbox_outside_weights = (bbox_inside_weights > 0).float()

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):

        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(
            batch_size, rois_per_image, 4 * num_classes).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        #print(bbox_target_data.shape, bbox_targets.shape, bbox_inside_weights.shape)

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).reshape(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                cls = clss[b][ind]
                start = int(cls * 4)
                end = start + 4
                bbox_targets[b, ind, start:end] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, start:end] = self.inside_weights

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        # cfg = self.cfg

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        targets = bbox_transform_batch(ex_rois, gt_rois)
        # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - self.means.expand_as(targets)) / self.stds.expand_as(targets))
        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, num_classes):

        # cfg = self.cfg
        fg_rois_per_image = self.batch_size * self.fg_thresh

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.reshape(-1, 1).type_as(gt_assignment) + gt_assignment

        # changed indexing way for pytorch 1.0
        labels = gt_boxes[:, :, 4].reshape(-1)[(offset.reshape(-1),)].reshape(batch_size, -1)

        labels_batch = labels.new(batch_size, self.batch_size).zero_()
        rois_batch = all_rois.new(batch_size, self.batch_size, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, self.batch_size, 5).zero_()
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= self.fg_thresh).reshape(-1)
            fg_num_rois = fg_inds.numel()
            bg_inds = torch.nonzero((max_overlaps[i] < self.bg_thresh_hi) &
                                    (max_overlaps[i] >= self.bg_thresh_lo)).reshape(-1)
            bg_num_rois = bg_inds.numel()
            # print("=> fg_num_rois: ", fg_num_rois,
            #   " bg_num_rois: ", bg_num_rois)

            if fg_num_rois > 0 and bg_num_rois > 0:

                fg_rois_per_this_image = int(min(fg_rois_per_image, fg_num_rois))
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = self.batch_size - fg_rois_per_this_image
                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(
                    bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(self.batch_size) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = self.batch_size
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(self.batch_size) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = self.batch_size
                fg_rois_per_this_image = 0
            else:
                bg_inds = torch.nonzero(max_overlaps[i] == -1).reshape(-1)
                rand_num = np.floor(np.random.rand(self.batch_size) * self.batch_size)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
                fg_rois_per_this_image = 0
                #raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < self.batch_size:
                labels_batch[i][fg_rois_per_this_image:] = 0
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
        bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(
                bbox_target_data,
                labels_batch,
                num_classes)
        # print(bbox_targets.shape)
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
