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
# from lib.roi_layers.nms import nms
from torchvision.ops import nms
from models.config import cfg
import cv2
from models.resnet import resnet
from tools.pascal_voc import write_bbndboxes, PascalVocWriter

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
#PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])
#PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])
thresh = 0.25
nms_thresh = 0.35
#class_list = ('__background__', 'SL', 'FC', 'YH', 'M')
class_list = ('__background__', 'SL', 'HH', 'FC', 'GG', 'QZ', 'M', 'TT', 'AJ', 'HM', 'PP')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--model', default = "./save_models/edge_sl_hh_0422_0225.ckpt",help='model')
  parser.add_argument('--img_path', default = "/data/datasets/edge-SL-HH-0422/original/",help='Path to containing images')
  parser.add_argument('--save_dir', default='./outputs/', type=str)
  parser.add_argument('--network', default='lite', type=str)
  parser.add_argument('--classes', default=5, type=int)
  parser.add_argument('--gama', default=False, type=bool)

  args = parser.parse_args()
  return args

def prepareTestData(target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE):
  batch_size = 1
  width, height, channles = get_target_size(target_size, im, 32, MAX_SIZE)
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
        filter_keep = _filter_boxes(bboxes_keep, 8)
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

def main():
  args = parse_args()
  img_path = args.img_path
  save_dir = args.save_dir
  args.classes = len(class_list)
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
  print(args.img_path)
  batch_size = 1
  std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
  mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
  std = torch.from_numpy(std).cuda()
  mean = torch.from_numpy(mean).cuda()
  
  for ix, img_name in enumerate(imgs):
    #print(ix, img_name)
    im = cv2.imread(img_name)
    name = os.path.basename(img_name)
    if im is None:
        continue
    import time
    #start_t = time.clock()
    if args.gama:
      im = adjust_gamma(im, 2.0)
    data = prepareTestData(640, im, 32, 1440)
    start_t = time.clock()
    results = im_detect(data, model, batch_size, std, mean, args.classes)
    end_t = time.clock() 
    print(ix, img_name, ' time consume is: ', end_t - start_t)
    draw_img = im.copy()
    save_xml = False
    if save_xml:
      imgFolderPath = os.path.dirname(img_name)
      imgFolderName = os.path.split(imgFolderPath)[-1]
      imgFileName = os.path.basename(img_name)
      image = cv2.imread(img_name)
      imageShape = [image.shape[0], image.shape[1], image.shape[2]]
      writer = PascalVocWriter(imgFolderName, imgFileName,
                                  imageShape, localImgPath=img_name)
      writer.verified = False
    save_flag = 0
    for boxes in results:
      for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        score = float(box[4])
        label = int(box[5])
        w = x2 - x1
        h = y2 - y1
        label_name = class_list[label]

        if label_name != 'HH':
            continue
        else:
            save_flag = 1
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (label * 50, label * 100, label * 100), 1)
        cv2.putText(draw_img, label_name, (x1 + w//2, y1 + h//2), cv2.FONT_HERSHEY_COMPLEX, 1, (label*50, label*100, label*100), 1)
        cv2.putText(draw_img, str(round(score, 3)), (x1 + w // 2 + 10, y1 + h // 2 + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (label * 50, label * 100, label * 100), 1)

        #if save_xml:
        #  if 'M' == label_name or 'SL' == label_name or 'HH' == label_name:
        #    continue
        #  difficult = 0
        #  writer.addBndBox(x1, y1, x2, y2, label_name, difficult)
      if save_xml:
          writer.save(targetFile='./outputs/' + name.replace('.jpg', '.xml'))
    if len(results) > 0 and save_flag:
        cv2.imwrite(save_dir +'/' + name, draw_img)


if __name__ == "__main__":
  main()
