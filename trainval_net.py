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
import collections
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from lib.datasets.pascal_voc import prepareBatchData

import os
import random

from lib.datasets.dataloader import CocoDataset, CSVDataset, XML_VOCDataset
from lib.datasets.dataloader import collater, Resizer, ResizerMultiScale, AspectRatioBasedSampler
from lib.datasets.dataloader import Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import lib.datasets.coco_eval as coco_eval
import lib.datasets.csv_eval as csv_eval
import lib.datasets.voc_eval as voc_eval

#from models.model import network
from models.config import cfg
from tools.net_utils import get_current_lr
from collections import OrderedDict
from tools.net_utils import adjust_learning_rate
from models.lite import lite_faster_rcnn
from models.pvanet import pva_net
from models.resnet import resnethyper, resnet, resnet_pva

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class_list = ('__background__', 'FC', 'MK', 'SX', 'SH', 'LS', 'JG', 'HH', 'YHA', 'ZF', 'KQ', 'WC', 'WQ', 'RK', 'KP', 'KZ', 'YHS', 'JD', 'KT', 'SL', 'DJ', 'SL', 'KQ')

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--dataset', default="xml",help='Dataset type, must be one of csv or coco.')
  parser.add_argument('--train_path', default = "/data/ssy/vertical-0522/train/", help='Path to containing train files')
  parser.add_argument('--val_path', default = "/data/ssy/vertical-0522/test/", help='Path to containing train files')
  parser.add_argument('--epochs', help='Number of epochs', type=int, default=5000)
  parser.add_argument('--batch_size', help='batch_size', default=32, type=int)
  parser.add_argument('--model_name', help='vertical_0522_', default=2, type=int)
  parser.add_argument('--network', default='lite', type=str)
  parser.add_argument('--classes', default=5, type=int)
  parser.add_argument('--save_dir', default='./save_models', type=str)
  parser.add_argument('--gpus', default=[1], type=list)
  parser.add_argument('--num_works', default=96, type=int)

  args = parser.parse_args()
  return args

def main():
  args = parse_args()

  if args.dataset == 'coco':

    if args.coco_path is None:
      raise ValueError('Must provide --coco_path when training on COCO,')

    dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

  elif args.dataset == 'csv':

    if args.csv_train is None:
      raise ValueError('Must provide --csv_train when training on COCO,')

    if args.csv_classes is None:
      raise ValueError('Must provide --csv_classes when training on COCO,')

    dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if args.csv_val is None:
      dataset_val = None
      print('No validation annotations provided.')
    else:
      dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes,
                               transform=transforms.Compose([Normalizer(), Resizer()]))

  elif args.dataset == 'xml':
    if args.train_path is None:
      raise ValueError('Must provide --voc_train when training on PASCAL VOC,')
    dataset_train = XML_VOCDataset(img_path=args.train_path,
                                   xml_path=args.train_path, class_list=class_list,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), ResizerMultiScale()]))

    if args.val_path is None:
      dataset_val = None
      print('No validation annotations provided.')
    else:
      dataset_val = XML_VOCDataset(img_path=args.val_path, xml_path=args.val_path,
                                   class_list=class_list, transform=transforms.Compose([Normalizer(), Resizer()]))

  else:
    raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

  sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
  dataloader_train = DataLoader(dataset_train, num_workers=args.num_works, collate_fn=collater, batch_sampler=sampler)

  if dataset_val is not None:
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

  print('Num training images: {}'.format(len(dataset_train)))
  best_map = 0
  is_best_map = False
  print("Training models...")
  args.classes = len(class_list)

  pretrained = True
  if 'lite' in args.network:
    pretrained = False
    model = lite_faster_rcnn(args.classes, pretrained=pretrained)
  if 'pva' in args.network:
    model = pva_net(args.classes, pretrained=False)
  if 'resnet' in args.network:
    model = resnet(args.classes, num_layers=101, pretrained=pretrained)
  if 'resnet_pva' in args.network:
    model = resnet_pva(args.classes, pretrained=True)
  model.create_architecture()
  device_id = [int(elem) for elem in args.gpus if elem != ',']
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

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True, mode="max")
  loss_hist = collections.deque(maxlen=1024)
  for epoch_num in range(args.epochs):

    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
      # print('iter num is: ', iter_num)
      try:
        optimizer.zero_grad()
        _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox = model(data['img'].cuda(), data['im_info'].cuda(), data['annot'].cuda())
        loss = (rpn_loss_cls.mean() + rpn_loss_bbox.mean() \
                + loss_cls.mean() + loss_bbox.mean())
        if bool(loss == 0):
          continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        loss_hist.append(float(loss))
        epoch_loss.append(float(loss))
        if len(device_id) > 1:
          rpn_loss_cls = rpn_loss_cls.mean().item()
          rpn_loss_bbox = rpn_loss_bbox.mean().item()
          loss_cls = loss_cls.mean().item()
          loss_bbox = loss_bbox.mean().item()
          loss = loss.mean().item()
        else:
          rpn_loss_cls = rpn_loss_cls.item()
          rpn_loss_bbox = rpn_loss_bbox.item()
          loss_cls = loss_cls.item()
          loss_bbox = loss_bbox.item()
          loss = loss.item()
        if iter_num % 200 == 0:
          print(
            'Epoch: {} | Iteration: {}/{} | loss: {:1.5f} | rpn bbox loss: {:1.5f} | rpn cls loss: {:1.5f} | bbox loss: {:1.5f} | cls loss: {:1.5f} | Running loss: {:1.5f}'.format(
            epoch_num, iter_num, len(dataset_train)//args.batch_size, float(loss), float(rpn_loss_bbox), float(rpn_loss_cls), float(loss_bbox), float(loss_cls), np.mean(loss_hist)))

        del rpn_loss_cls
        del rpn_loss_bbox
        del loss_bbox
        del loss_cls
      except Exception as e:
        print(e)
        continue

    if args.dataset == 'coco':

      print('Evaluating dataset')

      coco_eval.evaluate_coco(dataset_val, model)

    elif args.dataset == 'csv' and args.csv_val is not None:

      print('Evaluating dataset')

      mAP = csv_eval.evaluate(dataset_val, model)
    elif args.dataset == 'xml' and args.val_path is not None:

      print('Evaluating dataset')

      mAP = voc_eval.evaluate(dataset_val, model)

    try:
      is_best_map = mAP[0][0] > best_map
      best_map = max(mAP[0][0], best_map)
    except:
      pass
    if is_best_map:
      print("Get better map: ", best_map)
      torch.save({
        'epoch': epoch_num,
        'save_dir': args.save_dir,
        'state_dict': state_dict},
        os.path.join(args.save_dir, args.model_name + 'best_.ckpt'))
    else:
      print("Current map: ", best_map)
    scheduler.step(best_map)

    state_dict = model.module.state_dict()
    if epoch_num != 0 and epoch_num % 5 == 0:
      torch.save({
        'epoch': epoch_num,
        'save_dir': args.save_dir,
        'state_dict': state_dict},
        os.path.join(args.save_dir, args.model_name + '%04d.ckpt' % epoch_num))

if __name__ == "__main__":
  main()


