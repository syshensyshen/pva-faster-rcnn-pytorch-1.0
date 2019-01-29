'''
author: syshen
date: 2019/01/28
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
from lib.datasets.pascal_voc import prepareBatchData
import os
from models.model import network
from models.config import cfg
from tools.net_utils import get_current_lr
from collections import OrderedDict
from tools.net_utils import adjust_learning_rate
from lib.datasets.pascal_voc import get_target_size
from lib.datasets.pascal_voc import im_list_to_blob
import cv2

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--model', default = "./save_models/kpt_040.ckpt",help='model')
  parser.add_argument('--img_path', default = "/data/ssy/front_parts/VOC2007/JPEGImages/",help='Path to containing images')
  parser.add_argument('--save_dir', default='./', type=str)

  args = parser.parse_args()
  return args

def prepareTestData(target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE):
  batch_size = 1
  width, height, channles = get_target_size(608, im, 32, 1440)
  im_scales = np.zeros((batch_size, 4), dtype = np.float32)
  gt_boxes = np.zeros((batch_size, 1, 5), dtype=np.float32)
  im_blobs = np.zeros((batch_size, channles, height, width), dtype = np.float32)  
  im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
  im_scale_x = width / float(im.shape[1])
  im_scale_y = height / float(im.shape[0])
  im_scale = np.array([np.hstack((height, width, im_scale_x, im_scale_y))], dtype=np.float32)
  im = im.astype(np.float32) / 255.0
  im = (im - PIXEL_MEANS) / PIXEL_STDS
  im_blob = im_list_to_blob(im)
  im_blobs[0,:,:,:] =im_blob 
  im_scales[0,:] = im_scale
  return [gt_boxes, im_blobs, im_scales]

def main():
  args = parse_args()
  img_path = args.img_path
  save_dir = args.save_dir
  if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

  model = network(5, align=True)
  checkpoint = torch.load(args.model) 
  model.load_state_dict(checkpoint['state_dict'])
  im_blobs = torch.FloatTensor(1).cuda()
  im_info = torch.FloatTensor(1).cuda()
  gt_boxes = torch.FloatTensor(1).cuda()
  im_blobs = Variable(im_blobs)
  im_info = Variable(im_info)
  gt_boxes = Variable(gt_boxes)

  model = torch.nn.DataParallel(model).cuda()
  model.eval()
  imgs = glob(os.path.join(args.img_path, "*.bmp"))  
  print(args.img_path)
  batch_size = 1
  
  for ix, img_name in enumerate(imgs):
        im = cv2.imread(img_name)
        data = prepareTestData(608, im, 32, 1440)
        gt_tensor = torch.from_numpy(data[0])
        im_blobs_tensor = torch.from_numpy(data[1])
        im_info_tensor = torch.from_numpy(data[2])
        #print(im_blobs_tensor.shape)
        gt_boxes.data.resize_(gt_tensor.size()).copy_(gt_tensor)
        im_blobs.data.resize_(im_blobs_tensor.size()).copy_(im_blobs_tensor)
        im_info.data.resize_(im_info_tensor.size()).copy_(im_info_tensor)
        rois, cls_prob, bbox_pred, _, _, _, _ = model(im_blobs, im_info, gt_boxes)
        #scores = cls_prob.data
        #boxes = rois.data[:, :, 1:5]
        print(rois, cls_prob, bbox_pred)
  

if __name__ == "__main__":
  main()