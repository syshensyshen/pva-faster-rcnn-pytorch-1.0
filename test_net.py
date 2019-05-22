'''
author: syshen
date: 2019/01/28-01/30
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
from lib.datasets.pascal_voc import load_pascal_annotation

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
#PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])
#PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])
thresh = 0.01
nms_thresh = 0.7
#classes = 9
def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--model', default = "./save_models/phone__019.ckpt")
  parser.add_argument('--img_path', default = "/data/ssy/front_parts/VOC2007/JPEGImages/")
  parser.add_argument('--xml_path', default = "/data/ssy/front_parts/VOC2007/Annotations/")
  parser.add_argument('--save_dir', default='./', type=str)
  parser.add_argument('--network', default='lite', type=str)
  parser.add_argument('--classes', default=21, type=int)

  args = parser.parse_args()
  return args

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

def im_detect(data, model, batch_size, std, mean, classes):
  gt_tensor = torch.autograd.Variable(torch.from_numpy(data[0]))
  im_blobs_tensor = torch.autograd.Variable(torch.from_numpy(data[1]))
  im_info_tensor = torch.autograd.Variable(torch.from_numpy(data[2]))
  #print(im_info_tensor)
  results = []
  with torch.no_grad():
    rois, cls_prob, bbox_pred = model(im_blobs_tensor.cuda(), \
                                                  im_info_tensor.cuda(), \
                                                  gt_tensor.cuda())
    #print(rois[:,:,1:5])                                              
    pred_boxes = bbox_transform_inv(rois[:,:,1:5], bbox_pred, batch_size, std, mean)
    pred_boxes = clip_boxes(pred_boxes, im_info_tensor.data, 1)
    scores = cls_prob
    #results = []
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
          result [:,0:4] = bboxes_keep.cpu()
          result [:,4] = cls_keep.cpu()
          result [:,5] = index
          results.append(result)
  return results

def get_all_annots(xml_path):
  xmls = glob(xml_path + '/*.xml')
  annots = []
  for xml in xmls:    
    xml_context = load_pascal_annotation(xml)
    annot = np.zeros((xml_context['boxes'].shape[0], 5))
    annot[:,0:4] = xml_context['boxes']
    annot[:,4:5] = xml_context['gt_classes'].reshape((xml_context['gt_classes'].shape[0], 1))
    annots.append(annot)
  return annots

def classed_detections(all_detections, classes):
  dict_ = {}
  for i in range(classes):
    dict_[str(i)] = np.zeros((0, 4))
  for results in all_detections:
    for boxes in results:
      for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        label = int(box[5])
        dict_[str(label)] = np.row_stack((dict_[str(label)], np.array([x1, x2, y1, y2])))
  return dict_

def classed_annots(annots, classes):
  dict_ = {}
  for i in range(classes):
    dict_[str(i)] = np.zeros((0, 4))
  for annot in annots:
    for gt in annot:
      label = int(gt[4])
      dict_[str(label)] = np.row_stack((dict_[str(label)], gt[0:4]))
  return dict_

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0])
    ih = np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[2] - a[0]) * (a[3] - a[1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def main():
  args = parse_args()
  img_path = args.img_path
  save_dir = args.save_dir
  if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
  
  if 'lite' in args.network:
    model = lite_faster_rcnn(args.classes)
  elif 'pva' in args.network:
    model = pva_net(args.classes)
  elif 'resnet' in args.network:
      model = resnet(args.classes, num_layers=101)

  checkpoint = torch.load(args.model)
  model.create_architecture()
  model.load_state_dict(checkpoint['state_dict'])  
  model = model.cuda()

  model = torch.nn.DataParallel(model).cuda()
  model.eval()
  imgs = glob(os.path.join(args.img_path, "*.jpg"))  
  #print(args.img_path)
  batch_size = 1
  std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
  mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
  std = torch.from_numpy(std).cuda()
  mean = torch.from_numpy(mean).cuda()
  all_detections = []
  for ix, img_name in enumerate(imgs):
    #print(ix, img_name)
    im = cv2.imread(img_name)
    if im is None:
        continue
    import time
    #start_t = time.clock()
    data = prepareTestData(640, im, 32, 1440)
    start_t = time.clock()
    results = im_detect(data, model, batch_size, std, mean, args.classes)
    end_t = time.clock() 
    print(ix, img_name, ' time consume is: ', end_t - start_t)
    #results = np.array(results)
    all_detections.append(results)
  dict_ = classed_detections(all_detections, args.classes)
  annots = get_all_annots(args.xml_path)
  annots_ = classed_annots(annots, args.classes)

  for i in range(1, args.classes):    
    predictions = dict_[str(i)]
    annot_ = annots_[str(i)]
    false_positives = 0
    true_positives  = 0
    num_annotations = annot_.shape[0]
    for annot in annot_:
      overlaps            = compute_overlap(annot, predictions)
      if predictions.shape[0] == 0 or overlaps.shape[0] == 0:
        false_positives += 1
        continue
      assigned_annotation = np.argmax(overlaps)
      max_overlap         = overlaps[assigned_annotation]
      if max_overlap >= 0.01:
        #false_positives = np.append(false_positives, 0)
        true_positives  += 1
      else:
        false_positives += 1
        #true_positives  = np.append(true_positives, 0)
    #print(annot_.shape)
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    print('label ', str(i), ' ap is: ', recall, precision)


if __name__ == "__main__":
  main()


