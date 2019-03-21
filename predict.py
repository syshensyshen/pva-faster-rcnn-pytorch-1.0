'''
author: syshen
date: 2019/02/20
'''
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from glob import glob
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from lib.rpn.bbox_transform import clip_boxes
from lib.datasets.pascal_voc import prepareBatchData
import os
from models.lite import lite_faster_rcnn
from models.pvanet import pva_net
from models.config import cfg
from tools.net_utils import get_current_lr
from collections import OrderedDict
from tools.net_utils import adjust_learning_rate
from lib.datasets.pascal_voc import get_target_size
from lib.datasets.pascal_voc import im_list_to_blob
from lib.roi_layers.nms import nms
from models.config import cfg
import cv2
from models.resnet import resnet

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])
class_list = ('__background__', 'object_1', 'object_2', 'object_3', 'object_4', 'object_5', \
              'object_6', 'object_7','object_8', 'object_9', 'object_10', 'object_11', 'object_12', \
              'object_13', 'object_14', 'object_15', 'object_16', 'object_17', 'object_18', 'object_19', \
              'object_20', 'object_21', 'object_22', 'object_23', 'object_24', 'object_25', 'object_26', \
              'object_27', 'object_28', 'object_29', 'object_30', 'object_31', 'object_32' \
              'object_33', 'object_34', 'object_35', 'object_36', 'object_37', 'object_38', \
              'object_39', 'object_40', 'object_41', 'object_42', 'object_43', 'object_44', \
              'object_45')#, 'object_46', 'object_47','object_48', 'object_49')


def prepareTestData(target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE):
  batch_size = 1
  width, height, channles = get_target_size(736, im, 32, 1440)
  im_scales = np.zeros((batch_size, 4), dtype = np.float32)
  gt_boxes = np.zeros((batch_size, 1, 5), dtype=np.float32)
  im_blobs = np.zeros((batch_size, channles, height, width), dtype = np.float32)  
  im_scale_x = float(width) / float(im.shape[1])
  im_scale_y = float(height) / float(im.shape[0])
  im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)  
  im_scale = np.array([np.hstack((height, width, im_scale_x, im_scale_y))], dtype=np.float32)
  im = im.astype(np.float32) / 255
  im = (im - PIXEL_MEANS) / PIXEL_STDS
  im_blob = im_list_to_blob(im)
  im_blobs[0,:,:,:] =im_blob 
  im_scales[0,:] = im_scale
  return [gt_boxes, im_blobs, im_scales]

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = ((ws >= min_size) & (hs >= min_size))
    return keep

def bbox_transform_inv(boxes, deltas, batch_size, std, mean):
    #print(boxes)
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4] * std[0] + mean[0]
    dy = deltas[:, :, 1::4] * std[1] + mean[1]
    dw = deltas[:, :, 2::4] * std[2] + mean[2]
    dh = deltas[:, :, 3::4] * std[3] + mean[3]
    #print(dx, dy, dw, dh)

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def im_detect(data, model, batch_size, thresh=0.8, nms_thresh=0.25, classes=2):
  gt_tensor = torch.autograd.Variable(torch.from_numpy(data[0]))
  im_blobs_tensor = torch.autograd.Variable(torch.from_numpy(data[1]))
  im_info_tensor = torch.autograd.Variable(torch.from_numpy(data[2]))
  #print(im_info_tensor)
  std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
  mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
  std = torch.from_numpy(std).cuda()
  mean = torch.from_numpy(mean).cuda()
  with torch.no_grad():
    rois, cls_prob, bbox_pred = model(im_blobs_tensor.cuda(), \
                                                  im_info_tensor.cuda(), \
                                                  gt_tensor.cuda())
    #print(rois[:,:,1:5])                                              
    pred_boxes = bbox_transform_inv(rois[:,:,1:5], bbox_pred, batch_size, std, mean)
    pred_boxes = clip_boxes(pred_boxes, im_info_tensor.data, 1)
    scores = cls_prob
    results = []
    #print(rois.shape, scores.shape, rois.shape, bbox_pred.shape, classes)
    for index in range(1, classes):
        cls_scores = scores[0,:,index]
        scores_over_thresh = (cls_scores > thresh)
        cls_keep = cls_scores[scores_over_thresh]
        bboxes_keep = pred_boxes[0,scores_over_thresh,index*4:(index+1)*4]
        filter_keep = _filter_boxes(bboxes_keep, 16)
        cls_keep = cls_keep[filter_keep]
        bboxes_keep = bboxes_keep[filter_keep,:]
        keep_idx_i = nms(bboxes_keep, cls_keep, nms_thresh)
        keep_idx_i = keep_idx_i.long().view(-1)
        bboxes_keep = bboxes_keep[keep_idx_i, :]
        cls_keep = cls_keep[keep_idx_i]
        bboxes_keep[:,0] /= im_info_tensor[0,2]
        bboxes_keep[:,1] /= im_info_tensor[0,3]
        bboxes_keep[:,2] /= im_info_tensor[0,2]
        bboxes_keep[:,3] /= im_info_tensor[0,3]
        if bboxes_keep.size(0) > 0:
          result = np.zeros((bboxes_keep.size(0), 6), dtype=np.float32)
          result[:,0:4] = bboxes_keep.cpu()
          result[:,4] = cls_keep.cpu()
          result[:,5] = index
          results.append(reslut)
  return results

def init_model(model_path, num_class, model_name):
    if 'lite' in model_name:
        model = lite_faster_rcnn(num_class)
    elif 'pva' in model_name:
        model = pva_net(num_class)
    elif 'resnet' in model_name:
        model = resnet(num_class, num_layers=101)
    #model = resnet(num_class, num_layers=101)
    checkpoint = torch.load(model_path)
    model.create_architecture()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    return model

def predict(model, img, batch_size=1, thresh=0.6, nms_thresh=0.25, classes=9):
    data = prepareBatchData(608, img, 32, 1440)
    results = im_detect(data, model, batch_size, thresh, nms_thresh, classes)
    return results

def predict(model, img, batch_size=1, thresh=0.6, nms_thresh=0.25, classes=9):
  data = prepareBatchData(608, img, 32, 1440)
  results = im_detect(data, model, batch_size, thresh, nms_thresh, classes)
  predict_ = []
  for result in results:
    bbox = result[:,0:4]
    #score = result[:,4]
    index = result[:,5]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    dict_ = {}
    dict_['label'] = class_list[index]
    dict_['left'] = x1
    dict_['top'] = y1
    dict_['right'] = x2
    dict_['bottom'] = y2
    predict_.append(dict_)
  return predict_
