# -*- coding: utf-8 -*-
'''
# @Author  : syshen
# @date    : 2019/1/16 18:19
# @File    : region regression
'''
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proposal_layer import ProposalLayer
from .anchor_target_layer import AnchorTargetLayer
from models.smoothl1loss import _smooth_l1_loss

import numpy as np

from ..backbone.basic_modules import kaiming_init, normal_init
from .bbox_transform import bbox_overlaps, bbox_transform
from .anchor_tools import Anchor
from torchvision.ops import nms


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

def anchor_target_perimg(all_anchors,
                         gt_bboxes,
                         inside_flags,
                         inside_weight,
                         pos_overlap,
                         neg_overlap,
                         batch_size,
                         fraction):
    device = all_anchors.device
    anchors = all_anchors[inside_flags, :]
    num_anchors = all_anchors.size(0)
    num_valid_anchors = anchors.size(0)

    overlap = bbox_overlaps(anchors, gt_bboxes[:, 0:4])
    max_overlaps, argmax_overlaps = torch.max(overlap, 1)
    gt_max_overlaps, _ = torch.max(overlap, 0)

    bbox_targets = torch.zeros_like(anchors)
    bbox_inside_weight = torch.zeros_like(anchors)
    bbox_outside_weight = torch.zeros_like(anchors)
    label = torch.zeros(num_valid_anchors, dtype=torch.long).fill_(-1).to(device)

    label[max_overlaps < neg_overlap] = 0
    gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
    keep = torch.sum(overlap.eq(gt_max_overlaps.view(1, -1).expand_as(overlap)), 1)

    if torch.sum(keep) > 0:
        label[keep > 0] = 1

    label[max_overlaps >= pos_overlap] = 1

    num_fg = int(fraction * batch_size)

    sum_fg = torch.sum((label == 1).int(), 0)
    sum_bg = torch.sum((label == 0).int(), 0)

    if sum_fg > num_fg:
        fg_inds = torch.nonzero(label == 1).reshape(-1)
        rand_num = torch.randperm(fg_inds.size(0)).to(device).long()
        disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
        label[disable_inds] = -1
        sum_fg = num_fg

    num_bg = batch_size - sum_fg

    if sum_bg > num_bg:
        bg_inds = torch.nonzero(label == 0).reshape(-1)
        rand_num = torch.randperm(bg_inds.size(0)).to(device).long()
        disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
        label[disable_inds] = -1
        sum_bg = num_bg

    bbox_target = bbox_transform(anchors, gt_bboxes[argmax_overlaps.reshape(-1), :].reshape(-1, 5))

    bbox_inside_weight[label == 1] = inside_weight

    num_examples = torch.sum(label >= 0)
    positive_weights = 1.0 / num_examples.item()
    negative_weights = 1.0 / num_examples.item()
    bbox_outside_weight[label == 1] = positive_weights
    bbox_outside_weight[label == 0] = negative_weights

    label = unmap(label, num_anchors, inside_flags, -1)
    bbox_target = unmap(bbox_target, num_anchors, inside_flags)
    bbox_inside_weight = unmap(bbox_inside_weight, num_anchors, inside_flags)
    bbox_outside_weight = unmap(bbox_outside_weight, num_anchors, inside_flags)

    return label, bbox_target, bbox_inside_weight, bbox_outside_weight

def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets

class RPNHead(nn.Module):

    def __init__(self, inchannels, num_anchors, use_sigmoid):
        cls_out_channels = 2
        if use_sigmoid:
            cls_out_channels = 1
        super(RPNHead, self).__init__()
        self.logit = nn.Conv2d(inchannels, num_anchors * cls_out_channels, 1)
        self.pred = nn.Conv2d(inchannels, num_anchors * 4, 1)

        self.__init_weight__()

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def __init_weight__(self):
        normal_init(self.logit, std=0.01)
        normal_init(self.pred, std=0.01)

    def forward(self, feats):
        logits = []
        preds = []
        for i in range(len(feats)):
            logits.append(self.logit(feats[i]))
            preds.append(self.pred(feats[i]))

        return logits, preds

class RPNModule(nn.Module):

    """ region proposal network """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.inchannels = cfg.inchannels
        self.output_channels = cfg.output_channels
        self.anchor_scales = cfg.scales
        self.anchor_ratios = cfg.ratios
        self.feat_stride = cfg.stride

        self.use_sigmoid = cfg.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = 1
        else:
            self.cls_out_channels = 2

        self.pre_nms = cfg.pre_nms
        self.post_nms = cfg.post_nms

        self.batch_size = cfg.batch_size
        self.neg_overlap = cfg.neg_thresh
        self.pos_overlap = cfg.pos_thresh
        self.pos_weight = cfg.pos_weight
        self.fraction = cfg.fraction
        self.inside_weight = cfg.inside_weight
        self.nms_thresh = cfg.nms_thresh

        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        self.rpn_ops = RPNHead(self.inchannels, self.num_anchors, self.use_sigmoid)

        # define anchor genarator layer
        self.anchors = []
        for stride in self.feat_stride:
            self.anchors.append(Anchor(stride, self.anchor_scales, self.anchor_ratios))

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

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          all_anchors,
                          img_shape):
        multi_levels = len(cls_scores)
        proposals = []
        scores = []
        for lvl in range(multi_levels):
            with torch.no_grad():
                cls_score = cls_scores[lvl]
                bbox_pred = bbox_preds[lvl]
                anchors = all_anchors[lvl]
                cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                if self.use_sigmoid:
                    score = cls_score.sigmoid()
                else:
                    score = cls_score.softmax(-1)[:, 1]
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                proposal = self.bbox_transform_inv(anchors, bbox_pred)
                proposal = self.clip_boxes(proposal, img_shape)
                min_size = min(self.feat_stride)
                keep = self._filter_boxes(proposal, min_size, img_shape)
                proposal = proposal[keep]
                score = score[keep]
                _, order = torch.sort(score, 0, True)
                pre_nms = self.pre_nms // self.feat_stride[lvl]
                if pre_nms < score.size(0):
                    order = order[:pre_nms]

                proposal = proposal[order]
                score = score[order].reshape(-1)

                keep_idx = nms(proposal, score, self.nms_thresh).long()

                post_nms = self.post_nms // self.feat_stride[lvl]
                if post_nms < keep_idx.size(0):
                    keep_idx = keep_idx[:post_nms]
                proposal = proposal[keep_idx]
                score = score[keep_idx]

                proposals.append(proposal)
                scores.append(score)

        proposals = torch.cat(proposals)
        scores = torch.cat(scores)

        return proposals, scores

    def proposal(self, anchors, cls_scores, bbox_preds, im_info):
        batch_size = len(im_info)
        proposals = []
        num_levels = len(cls_scores)
        proposals = torch.zeros((batch_size, self.post_nms, 5)).to(cls_scores[0].device)
        for img_id in range(batch_size):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            proposal, score = self.get_bboxes_single(cls_score_list, bbox_pred_list, anchors, im_info[img_id])
            num_proposal = proposal.size(0)
            proposals[img_id, :, 0] = img_id
            proposals[img_id, :num_proposal, 1:] = proposal

        return proposals

    def get_targets(self, all_anchors, all_inside_flags, im_info, gt_boxes):
        img_per_img = im_info.size(0)
        labels = []
        bbox_targets = []
        bbox_inside_weights = []
        bbox_outside_weights = []
        for i in range(img_per_img):
            label, bbox_target, bbox_inside_weight, bbox_outside_weight = \
                anchor_target_perimg(all_anchors, gt_boxes[i], all_inside_flags, self.inside_weight, self.pos_overlap,
                                     self.neg_overlap, self.batch_size, self.fraction)
            labels.append(label)
            bbox_targets.append(bbox_target)
            bbox_inside_weights.append(bbox_inside_weight)
            bbox_outside_weights.append(bbox_outside_weight)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def loss(self, scores, bbox_preds, num_level_anchors, all_anchors, all_inside_flags, im_info, gt_boxes, loss_type="smoothL1loss"):
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self.get_targets(all_anchors, all_inside_flags, im_info, gt_boxes)

        labels = images_to_levels(labels, num_level_anchors)
        bbox_targets = images_to_levels(bbox_targets, num_level_anchors)
        bbox_inside_weights = images_to_levels(bbox_inside_weights, num_level_anchors)
        bbox_outside_weights = images_to_levels(bbox_outside_weights, num_level_anchors)

        loss_cls = 0.0
        loss_bbox = 0.0
        scale = 1.0
        for i in range(len(labels)):
            label = labels[i]
            bbox_target = bbox_targets[i]
            bbox_inside_weight = bbox_inside_weights[i]
            bbox_outside_weight = bbox_outside_weights[i]

            label = label.reshape(-1)
            cls_score = scores[i]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

            # rpn_keep = label.ne(-1).nonzero().reshape(-1).long()
            # cls_score = torch.index_select(cls_score, 0, rpn_keep)
            # label = torch.index_select(label, 0, rpn_keep)
            reg_inds = label >= 0
            cls_score = cls_score[reg_inds]
            label = label[reg_inds]
            if label.size(0) <= 0:
                # print('lvl is: ', i)
                # print('gt boxes is: ', gt_boxes)
                continue
            scale += 1
            if self.use_sigmoid:
                loss_cls += F.binary_cross_entropy(cls_score, label)
            else:
                loss_cls += F.cross_entropy(cls_score, label)

            bbox_pred = bbox_preds[i].permute(0, 2, 3, 1).reshape(-1, 4)
            bbox_target = bbox_target.reshape(-1, 4)
            bbox_inside_weight = bbox_inside_weight.reshape(-1, 4)
            bbox_outside_weight = bbox_outside_weight.reshape(-1, 4)

            bbox_pred = bbox_pred[reg_inds]
            bbox_target = bbox_target[reg_inds]
            bbox_inside_weight = bbox_inside_weight[reg_inds]
            bbox_outside_weight = bbox_outside_weight[reg_inds]

            loss_bbox += _smooth_l1_loss(bbox_pred, bbox_target, bbox_inside_weight, bbox_outside_weight, sigma=3, dim=[1])

        return loss_cls / scale, loss_bbox / scale

    def valid_flags(self, all_anchors, im_info):
        total_anchors = all_anchors.size(0)
        _allowed_border = 0
        keep = ((all_anchors[:, 0] >= -_allowed_border) &
                (all_anchors[:, 1] >= -_allowed_border) &
                (all_anchors[:, 2] < int(im_info[1]) + _allowed_border) &
                (all_anchors[:, 3] < int(im_info[0]) + _allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        return inds_inside

    def forward(self, feats, im_info, gt_boxes=None):
        cls_scores, bbox_preds = self.rpn_ops(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_anchors = []
        # all_inside_flags = []
        for i in range(len(feats)):
            anchors = self.anchors[i].grid_anchors(featmap_sizes[i], self.feat_stride[i], device=feats[i].device)
            # inside_flags = self.anchors[i].valid_flags(anchors, im_info[0])
            all_anchors.append(anchors)
            # all_inside_flags.append(inside_flags)
        num_level_anchors = [anchor.size(0) for anchor in all_anchors]

        proposals = self.proposal(all_anchors, cls_scores, bbox_preds, im_info)

        all_anchors = torch.cat(all_anchors)

        if self.training:
            all_inside_flags = self.valid_flags(all_anchors, im_info[0])
            # all_inside_flags = torch.cat(all_inside_flags)
            loss_cls, loss_bbox = self.loss(cls_scores, bbox_preds, num_level_anchors, all_anchors, all_inside_flags, im_info, gt_boxes)
            return proposals, loss_cls, loss_bbox
        else:
            return proposals


def get_model(config):
    return RPNModule(config)