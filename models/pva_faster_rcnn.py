# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# # from models.config import cfg
# from lib.rpn.rpn_regression import _RPN
# from collections import OrderedDict
# # from lib.prroi_pool.functional import prroi_pool2d
# from torchvision.ops import RoIPool as ROIPoolingLayer
# from torchvision.ops import RoIAlign as ROIAlignLayer
# from models.rcnn_tools.proposal_target_layer import ProposalTargetLayer
# from models.smoothl1loss import _smooth_l1_loss
#
# # added by Henson
# from models.giou import compute_iou
#
# from importlib import import_module
#
#
# class pva_faster_rcnn(nn.Module):
#     def freeze_bn(self):
#         '''Freeze BatchNorm layers.'''
#         for layer in self.modules():
#             if isinstance(layer, nn.BatchNorm2d):
#                 layer.eval()
#     """ faster RCNN """
#
#     def __init__(self, cfg, classes, class_agnostic):
#         super(pva_faster_rcnn, self).__init__()
#
#         self.cfg = cfg
#         self.n_classes = classes
#         self.class_agnostic = class_agnostic
#         # loss
#         self.RCNN_loss_cls = 0
#         self.RCNN_loss_bbox = 0
#         self.rcnn_din = cfg.MODEL.RCNN_CIN
#         self.rcnn_last_din = cfg.MODEL.RCNN_LAST
#         self.rpn_din = cfg.MODEL.RPN_CIN
#         self.topk = cfg.TRAIN.topk
#
#         # define rpn
#         self.rpn_tools = _RPN(cfg, self.dout_base_model, self.rpn_din)
#
#         # rcnn regression
#         mean = cfg.TRAIN.BBOX_NORMALIZE_MEANS
#         std = cfg.TRAIN.BBOX_NORMALIZE_STDS
#         inside_weight = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
#         fraction = cfg.TRAIN.FG_FRACTION
#         batch_size = cfg.TRAIN.BATCH_SIZE
#         fg_thresh = cfg.TRAIN.FG_THRESH
#         bg_thresh_hi = cfg.TRAIN.BG_THRESH_HI
#         bg_thresh_lo = cfg.TRAIN.BG_THRESH_LO
#         self.proposal_target = ProposalTargetLayer(mean, std, inside_weight, batch_size, fraction, self.n_classes, fg_thresh, bg_thresh_hi, bg_thresh_lo)
#
#         if cfg.POOLING_MODE == 'align':
#             self.rcnn_roi_pool = ROIAlignLayer(
#                 (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / cfg.FEAT_STRIDE[0], 2)
#         elif cfg.POOLING_MODE == 'pool':
#             self.rcnn_roi_pool = ROIPoolingLayer(
#                 (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / cfg.FEAT_STRIDE[0])
#         elif cfg.POOLING_MODE == 'prroi':
#             self.rcnn_roi_pool = prroi_pool2d(
#                 (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / cfg.FEAT_STRIDE[0])
#
#         self.rcnn_top = nn.Sequential(OrderedDict([
#             ('fc6_new', nn.Linear(self.dout_base_model *
#                                   cfg.POOLING_SIZE * cfg.POOLING_SIZE, self.rcnn_din)),
#             ('fc6_relu', nn.ReLU(inplace=True)),
#             ('fc7_new', nn.Linear(self.rcnn_din, self.rcnn_last_din, bias=True)),
#             ('fc7_relu', nn.ReLU(inplace=True))
#         ]))
#
#     def forward(self, im_data, im_info, gt_boxes):
#         cfg = self.cfg
#
#         # print(im_data.shape, im_info)
#         batch_size = im_data.size(0)
#
#         im_info = im_info.data
#         gt_boxes = gt_boxes.data
#         # num_boxes = num_boxes.data
#
#         # feed image data to base model to obtain base feature map
#         base_feat = self.backbone(im_data)
#         # print(base_feat.shape)
#         # feed base feature map tp RPN to obtain rois
#         rois, rpn_loss_cls, rpn_loss_bbox = self.rpn_tools(base_feat, im_info, gt_boxes)
#
#         # if it is training phrase, then use ground trubut bboxes for refining
#         with torch.no_grad():
#             if self.training:
#                 roi_data = self.proposal_target(rois, gt_boxes)
#                 rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
#
#                 rois_label = Variable(rois_label.view(-1).long())
#                 rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
#                 rois_inside_ws = Variable(
#                     rois_inside_ws.view(-1, rois_inside_ws.size(2)))
#                 rois_outside_ws = Variable(
#                     rois_outside_ws.view(-1, rois_outside_ws.size(2)))
#             else:
#                 rois_label = None
#                 rois_target = None
#                 rois_inside_ws = None
#                 rois_outside_ws = None
#                 rpn_loss_cls = 0
#                 rpn_loss_bbox = 0
#
#         # rois = Variable(rois)
#         # do roi pooling based on predicted rois
#         pooled_feat = self.rcnn_roi_pool(base_feat, rois.view(-1, 5))
#
#         # feed pooled features to top model
#         pooled_feat = self._head_to_tail(pooled_feat)
#         # print(self.training)
#
#         # compute bbox offset
#         bbox_pred = self.rcnn_bbox_pred(pooled_feat)
#
#         # compute object classification probability
#         cls_score = self.rcnn_cls_score(pooled_feat)
#         cls_prob = F.softmax(cls_score, 1)
#
#         rcnn_loss_cls = 0
#         rcnn_loss_bbox = 0
#
#         if self.training:
#             # classification loss
#             if self.cfg.TRAIN.is_ohem_rcnn:
#                 rcnn_loss_cls = F.cross_entropy(
#                     cls_score, rois_label, reduction='none')
#
#                 top_k = self.topk
#                 _, topk_loss_inds = rcnn_loss_cls.topk(top_k)
#                 rcnn_loss_cls = rcnn_loss_cls[topk_loss_inds].mean()
#             else:
#                 rcnn_loss_cls = F.cross_entropy(
#                     cls_score, rois_label)
#
#             if cfg.TRAIN.loss_type == "smoothL1loss":
#                 if self.cfg.TRAIN.is_ohem_rcnn:
#                     rcnn_loss_bbox = _smooth_l1_loss(bbox_pred[topk_loss_inds, :], rois_target[topk_loss_inds, :],
#                                                      rois_inside_ws[topk_loss_inds, :], rois_outside_ws[topk_loss_inds, :], sigma=3.0)
#
#                     # RCNN_loss_bbox = _smooth_l1_loss(
#                     #     bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
#
#                 else:
#                     # bounding box regression L1 loss
#                     rcnn_loss_bbox = _smooth_l1_loss(
#                         bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
#
#                 # RCNN_loss_bbox = _balance_smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
#             elif "IOUloss" in cfg.TRAIN.loss_type:
#                 iou, g_iou = compute_iou(
#                     rois_target, rois_target, rois_inside_ws, rois_outside_ws)
#
#                 if cfg.TRAIN.loss_type == "GIOUloss":
#                     rcnn_loss_bbox = 1 - g_iou
#                 elif cfg.TRAIN.loss_type == "IOUloss":
#                     rcnn_loss_bbox = -iou.log()
#
#         cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
#         bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
#
#         if self.training:
#             # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
#             return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, rcnn_loss_cls, rcnn_loss_bbox
#         else:
#             return rois, cls_prob, bbox_pred
#
#     def _init_weights(self, cfg):
#         def normal_init(m, mean, stddev, truncated=False):
#             """
#             weight initalizer: truncated normal and random normal.
#             """
#             # x is a parameter
#             if truncated:
#                 m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
#                     mean)  # not a perfect approximation
#             else:
#                 m.weight.data.normal_(mean, stddev)
#                 m.bias.data.zero_()
#
#         normal_init(self.rpn_tools.rpn_preconv, 0, 0.01, cfg.TRAIN.TRUNCATED)
#         # normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
#         # normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
#         # normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
#         # normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
#         normal_init(self.rcnn_top.fc6_new, 0, 0.01, cfg.TRAIN.TRUNCATED)
#         normal_init(self.rcnn_top.fc7_new, 0, 0.01, cfg.TRAIN.TRUNCATED)
#
#         # nn.init.kaiming_normal_(self.RCNN_rpn.RPN_Conv.weight)
#         nn.init.kaiming_normal_(self.rpn_tools.cls_pred.weight)
#         nn.init.kaiming_normal_(self.rpn_tools.bbox_pred.weight)
#         nn.init.kaiming_normal_(self.rcnn_cls_score.weight)
#         nn.init.kaiming_normal_(self.rcnn_bbox_pred.weight)
#         # nn.init.kaiming_normal_(self.RCNN_top.fc6_new.weight)
#         # nn.init.kaiming_normal_(self.RCNN_top.fc7_new.weight)
#
#     def create_architecture(self, cfg):
#         self._init_modules(cfg)
#         self._init_weights(cfg)
#
#
# class faster_rcnn(nn.Module):
#     def __init__(self, cfg):
#         super(faster_rcnn, self).__init__()
#         self.backbone = import_module('backbone').get_model(cfg.MODEL.BACKBONE)
#     def forward(self, *input):
#         pass
