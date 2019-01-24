'''
author: syshen
date: 2019/01/22-01/23
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
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--dataset', default="csv",help='Dataset type, must be one of csv or coco.')
  parser.add_argument('--xml_path', default = "/data/ssy/front_parts/VOC2007/Annotations/",help='Path to containing annAnnotations')
  parser.add_argument('--img_path', default = "/data/ssy/front_parts/VOC2007/JPEGImages/",help='Path to containing images')
  parser.add_argument('--epochs', help='Number of epochs', type=int, default=600)
  parser.add_argument('--batch_size', help='batch_size', default=2, type=int)
  parser.add_argument('--save_dir', default='./', type=str)

  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  xml_path = args.xml_path
  img_path = args.img_path
  batch_size = args.batch_size
  save_dir = args.save_dir
  if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
  
  xmls = glob(os.path.join(args.xml_path, "*.xml"))
  sample_size = len(xmls)
  batch_szie = args.batch_size
  iters_per_epoch = int(np.floor(sample_size / batch_szie))
  model = network(3, align=True)
  #use_gpu = True
  #model = model.cuda()
  im_blobs = torch.FloatTensor(1).cuda()
  im_info = torch.FloatTensor(1).cuda()
  num_boxes = torch.LongTensor(1).cuda()
  gt_boxes = torch.FloatTensor(1).cuda()
  im_blobs = Variable(im_blobs)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  model = torch.nn.DataParallel(model).cuda()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=cfg.TRAIN.MOMENTUM)
  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True,mode="max")
  #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, patience=15, verbose=True,mode="max")
  for epoch in range(0, args.epochs):
    loss_temp = 0
    for iters in range(0, iters_per_epoch):
      optimizer.zero_grad()
      optimizer.step()
      #lr_scheduler.step()
      model.train()
      start_iter = iters * batch_szie
      end_iter = start_iter + batch_szie
      if end_iter > sample_size:
        end_iter = sample_size
        start_iter = end_iter - batch_szie + 1
      if end_iter == start_iter:
        start_iter = end_iter - 1
      xml = xmls[start_iter:end_iter]
      data = prepareBatchData(xml_path, img_path, batch_size, xml)
      gt_tensor = torch.from_numpy(data[0])
      im_blobs_tensor = torch.from_numpy(data[1])
      im_info_tensor = torch.from_numpy(data[2])
      #print(im_blobs_tensor.shape)
      gt_boxes.data.resize_(gt_tensor.size()).copy_(gt_tensor)
      im_blobs.data.resize_(im_blobs_tensor.size()).copy_(im_blobs_tensor)
      im_info.data.resize_(im_info_tensor.size()).copy_(im_info_tensor)
      #print(gt_boxes.shape)
      #print(im_blobs.shape)
      #print(im_scales.shape)
      rois, cls_prob, bbox_pred, rpn_loss_cls, \
      rpn_loss_bbox, loss_cls, loss_bbox = model(im_blobs, im_info, gt_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() \
           + loss_cls.mean() + loss_bbox.mean()
      loss_temp += loss.item()      
      loss.backward()      

      if iters % 100 == 0:
        if iters > 0:
          loss_temp /= 50   

        current_lr = get_current_lr(optimizer)   

        print("[epoch %2d][iter %4d/%4d] loss: %.4f lr: %.4f" \
                                % (epoch, iters, iters_per_epoch, loss_temp, current_lr))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox))
    state_dict = model.module.state_dict()
    if epoch >= 30 and epoch % 20 ==0:
      torch.save({
      'epoch': epoch,
      'save_dir': save_dir,
      'state_dict': state_dict},
      os.path.join(save_dir, 'kpt' + '_%03d.ckpt' % epoch))

if __name__ == "__main__":
  main()
