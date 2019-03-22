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
import random
#from models.model import network
from models.config import cfg
from tools.net_utils import get_current_lr
from collections import OrderedDict
from tools.net_utils import adjust_learning_rate
from models.lite import lite_faster_rcnn
from models.pvanet import pva_net
from models.resnet import resnethyper, resnet_pva
from models.resnet import resnet18, resnet34,resnet50, resnet101, resnet152
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--dataset', default="csv",help='Dataset type, must be one of csv or coco.')
  parser.add_argument('--xml_path', default = "/data/ssy/front_parts/VOC2007/Annotations/",help='Path to containing annAnnotations')
  parser.add_argument('--img_path', default = "/data/ssy/front_parts/VOC2007/JPEGImages/",help='Path to containing images')
  parser.add_argument('--epochs', help='Number of epochs', type=int, default=1000)
  parser.add_argument('--batch_size', help='batch_size', default=2, type=int)
  parser.add_argument('--sub_batch', help='sub_batch', default=2, type=int)
  parser.add_argument('--network', default='lite', type=str)
  parser.add_argument('--classes', default=21, type=int)
  parser.add_argument('--save_dir', default='./', type=str)
  parser.add_argument('--gpus', default=[0], type=list)
  parser.add_argument('--depth', default=101, type=int)

  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  xml_path = args.xml_path
  img_path = args.img_path
  batch_size = args.batch_size
  sub_batch = args.sub_batch
  save_dir = args.save_dir
  if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
  
  xmls = glob(os.path.join(args.xml_path, "*.xml"))
  sample_size = len(xmls)
  batch_szie = args.batch_size
  iters_per_epoch = int(np.floor(sample_size / batch_szie))
  pretrained = True
  if 'lite' in args.network:
    pretrained = False
    model = lite_faster_rcnn(args.classes, pretrained=pretrained)
  if 'pva' in args.network:
    model = pva_net(args.classes, pretrained=pretrained)
  if 'resnet_pva' in args.network:
    model = resnet_pva(args.classes, pretrained=True)
  if 'resnet_fpn' in args.network:
    model = eval('resnet'+str(args.depth))(args.classes, training=True, pretrained=True)
  if not 'resnet_fpn' in args.network:
      model.create_architecture()
  device_id = [ int(elem) for elem in args.gpus if elem != ',']
  if len(device_id) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_id)
  else:
    model = torch.nn.DataParallel(model)
  use_gpu = True
  if use_gpu:
    model = model.cuda()
  
  model.train()
  if pretrained and 'lite' in args.network:
    model.module.freeze_bn()
  #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0.00005)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00005)
  #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=60, verbose=True,mode="max")
  #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, patience=15, verbose=True,mode="max")
  #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
  #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[62, 102, 132, 145], gamma=0.1)
  milestones=[200, 500, 600, 1000]
  index = 0
  lr_decay_step = milestones[index] 

  for epoch in range(0, args.epochs):
    if epoch % 20 == 0 and epoch != 0:
        random.shuffle(xmls)
    loss_temp = 0    
    if epoch > lr_decay_step:
      index += 1
      lr_decay_step = milestones[index]
      adjust_learning_rate(optimizer, 0.1)

    for iters in range(0, iters_per_epoch):            
      start_iter = iters * batch_szie
      end_iter = start_iter + batch_szie
      if end_iter > sample_size:
        end_iter = sample_size
        start_iter = end_iter - batch_szie
      if end_iter == start_iter:
        start_iter = end_iter - 1
      optimizer.zero_grad()
      loss = 0
      xml = xmls[start_iter:end_iter]
      data = prepareBatchData(xml_path, img_path, batch_size, xml)
      gt_tensor = torch.autograd.Variable(torch.from_numpy(data[0]))
      im_blobs_tensor = torch.autograd.Variable(torch.from_numpy(data[1]))
      im_info_tensor = torch.autograd.Variable(torch.from_numpy(data[2]))
      if use_gpu:
        gt_tensor = gt_tensor.cuda()
        im_blobs_tensor = im_blobs_tensor.cuda()
        im_info_tensor = im_info_tensor.cuda()
      for sub in range(int(batch_size / sub_batch)):
        start_sub = int(sub * batch_size / sub_batch)
        end_sub = int((sub + 1) * batch_size / sub_batch)

        if pretrained and 'lite' in args.network:
           model.module.freeze_bn()      
        _, _, _, rpn_loss_cls, \
        rpn_loss_bbox, loss_cls, loss_bbox, _ = \
                        model(im_blobs_tensor[start_sub:end_sub, :, :, :], \
                        im_info_tensor[start_sub:end_sub, :], gt_tensor[start_sub:end_sub, :, :])

        loss += (rpn_loss_cls.mean() + rpn_loss_bbox.mean() \
            + loss_cls.mean() + loss_bbox.mean())
      loss_temp += loss.item()      
      loss.backward()
      optimizer.step()
      display = 20

      if iters % display == 0:
        if iters > 0:
          loss_temp /= display 

        if len(device_id) > 1:
          rpn_loss_cls = rpn_loss_cls.mean().item()
          rpn_loss_bbox = rpn_loss_bbox.mean().item()
          loss_cls = loss_cls.mean().item()
          loss_bbox = loss_bbox.mean().item()
        else:
          rpn_loss_cls = rpn_loss_cls.item()
          rpn_loss_bbox = rpn_loss_bbox.item()
          loss_cls = loss_cls.item()
          loss_bbox = loss_bbox.item()

        current_lr = get_current_lr(optimizer)   

        print("[epoch %2d][iter %4d/%4d] loss: %.4f lr: %.8f" \
                                % (epoch, iters, iters_per_epoch, loss_temp, current_lr))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox))
    state_dict = model.module.state_dict()
    if epoch!=0 and epoch%30==0:
      torch.save({
      'epoch': epoch,
      'save_dir': save_dir,
      'state_dict': state_dict},
      os.path.join(save_dir, 'hh_fpn_' + '%04d.ckpt' % epoch))

if __name__ == "__main__":
  main()
