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
# from lib.datasets.pascal_voc import prepareBatchData

import os
import random

from lib.datasets.dataloader import CocoDataset, CSVDataset, XML_VOCDataset
from lib.datasets.dataloader import collater, Resizer, ResizerMultiScale, AspectRatioBasedSampler
from lib.datasets.dataloader import Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from models.lite import lite_faster_rcnn
from models.pvanet import pva_net
from models.resnet import resnet, resnet_pva

# added by Henson
from lib.config.config import Config, merge_config
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from coco_mAP import calc_fscore


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')
    parser.add_argument(
        '--base_config', default="lib/config/base_detection_config.py", help='config')
    parser.add_argument(
        '--config', default="lib/config/det_whiteshow_config.py", help='config')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg.model_cfg, det_cfg.model_cfg)

    gpus = ",".join('%s' % id for id in cfg.TRAIN.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpu_nums = len(gpus.split(','))

    if cfg.TRAIN.dataset == 'coco':

        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Augmenter()]))
        dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Augmenter()]))

    elif cfg.TRAIN.dataset == 'csv':

        if args.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if args.csv_classes is None:
            raise ValueError(
                'Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if args.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    elif cfg.TRAIN.dataset == 'xml':
        if cfg.TRAIN.train_path is None:
            raise ValueError(
                'Must provide --voc_train when training on PASCAL VOC,')
        dataset_train = XML_VOCDataset(cfg, img_path=cfg.TRAIN.train_path,
                                       xml_path=cfg.TRAIN.train_path, class_list=cfg.class_list,
                                       transform=transforms.Compose([Augmenter()]))

        if cfg.TRAIN.val_path is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = XML_VOCDataset(cfg, img_path=cfg.TRAIN.val_path, xml_path=cfg.TRAIN.val_path,
                                         class_list=cfg.class_list, transform=transforms.Compose([]))

    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=cfg.TRAIN.batch_size_per_gpu * gpu_nums, drop_last=False)
    dataloader_train = DataLoader(
        dataset_train, num_workers=cfg.TRAIN.num_works, collate_fn=collater, batch_sampler=sampler, pin_memory=True)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(
            dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val, pin_memory=True)

    print('Num training images: {}'.format(len(dataset_train)))
    print("Training models...")
    classes = len(cfg.class_list)

    pretrained = True
    if 'lite' in cfg.MODEL.BACKBONE:
        pretrained = False
        model = lite_faster_rcnn(cfg, classes, pretrained=pretrained)
    if 'pva' in cfg.MODEL.BACKBONE:
        model = pva_net(classes, pretrained=False)
    # if 'resnet' in cfg.MODEL.BACKBONE:
    #     model = resnet(classes, num_layers=101, pretrained=pretrained)
    # if 'resnet_pva' in cfg.MODEL.BACKBONE:
    #     model = resnet_pva(classes, pretrained=True)
    # pretrained = True

    model.create_architecture(cfg)

    start_epoch = 1
    if cfg.TRAIN.resume_model is not None:  # added by Henson
        checkpoint = torch.load(cfg.TRAIN.resume_model)

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        start_epoch = int(checkpoint['epoch']) + 1

        print("loading pretrained model: ", cfg.TRAIN.resume_model)

    if gpu_nums > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        # cudnn.benchmark = True
        model = model.cuda()
    model.train()

    # if pretrained and not 'lite' in cfg.MODEL.BACKBONE:
    #   model.module.freeze_bn()

    learning_rate_base = cfg.TRAIN.LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, verbose=True, mode="max")
    loss_hist = collections.deque(maxlen=1024)
    # min_loss = 1.0
    min_avg_loss_hist = 110000.0
    for epoch_num in range(start_epoch, cfg.TRAIN.epochs):
        print("\n=> learning_rate: ", learning_rate_base,
              " min_avg_loss_hist: ", min_avg_loss_hist)

        # epoch_loss = []
        loss_hist.clear()
        # loss, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox = 0, 0, 0, 0, 0
        for iter_num, data in enumerate(dataloader_train):
            # print('iter num is: ', iter_num)
            # print("\n", data['im_info'])
            # continue
            try:
                optimizer.zero_grad()
                _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox = model(
                    data['img'].cuda(), data['im_info'].cuda(), data['annot'].cuda())
                # rpn_loss_bbox *= 15
                # rpn_loss_cls  *= 20
                # loss_bbox     *= 15
                # loss_cls      *= 20
                loss = (rpn_loss_cls.mean() + 1.0 * rpn_loss_bbox.mean()
                        + loss_cls.mean() + 1.0 * loss_bbox.mean())
                if bool(loss == 0):
                    continue
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                loss_hist.append(float(loss))
                # epoch_loss.append(float(loss))
                if gpu_nums > 1:
                    rpn_loss_cls = rpn_loss_cls.mean().item()
                    rpn_loss_bbox = 1.0 * rpn_loss_bbox.mean().item()
                    loss_cls = loss_cls.mean().item()
                    loss_bbox = 1.0 * loss_bbox.mean().item()
                    loss = loss.mean().item()
                else:
                    rpn_loss_cls = rpn_loss_cls.item()
                    rpn_loss_bbox = 1.0 * rpn_loss_bbox.item()
                    loss_cls = loss_cls.item()
                    loss_bbox = 1.0 * loss_bbox.item()
                    loss = loss.item()

                if iter_num % 20 == 0:
                    print(
                        'Epoch: {} | Iteration: {}/{} | loss: {:1.5f} | rpn bbox loss: {:1.5f} | rpn cls loss: {:1.5f} | bbox loss: {:1.5f} | cls loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, len(dataset_train)//(cfg.TRAIN.batch_size_per_gpu * gpu_nums), float(loss), float(rpn_loss_bbox), float(rpn_loss_cls), float(loss_bbox), float(loss_cls), np.mean(loss_hist)))

                del rpn_loss_cls
                del rpn_loss_bbox
                del loss_bbox
                del loss_cls
            except Exception as e:
                print('Epoch: {} | Iteration: {}/{} | Exception: {}'.format(epoch_num, iter_num, len(
                    dataset_train)//(cfg.TRAIN.batch_size_per_gpu * gpu_nums), e))
                continue

        # if cfg.TRAIN.dataset == 'coco':
        #
        #   print('Evaluating dataset')
        #
        #   coco_eval.evaluate_coco(dataset_val, model)
        #
        # elif cfg.TRAIN.dataset == 'csv' and args.csv_val is not None:
        #
        #   print('Evaluating dataset')
        #
        #   mAP = csv_eval.evaluate(dataset_val, model)
        # elif cfg.TRAIN.dataset == 'xml' and cfg.TRAIN.val_path is not None:
        #
        #   print('Evaluating dataset')
        #
        #   mAP = voc_eval.evaluate(dataset_val, model)
        #
        # try:
        #   is_best_map = mAP[0][0] > best_map
        #   best_map = max(mAP[0][0], best_map)
        # except:
        #   pass
        # if is_best_map:
        #   print("Get better map: ", best_map)
        #   torch.save({
        #     'epoch': epoch_num,
        #     'save_dir': args.save_dir,
        #     'state_dict': state_dict},
        #     os.path.join(args.save_dir, args.model_name + 'best_.ckpt'))
        # else:
        #   print("Current map: ", best_map)
        # scheduler.step(best_map)

        if epoch_num % cfg.save_model_interval == 0 and epoch_num > 0:
            if gpu_nums > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            learning_rate_base /= 10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)

            save_model_path = os.path.join(
                cfg.TRAIN.save_dir, cfg.TRAIN.model_name + '%04d.ckpt' % epoch_num)
            print("save model: ", save_model_path)
            torch.save({
                'epoch': epoch_num,
                'save_dir': cfg.TRAIN.save_dir,
                'state_dict': state_dict}, save_model_path)

        # if epoch_num % 10 == 0 and epoch_num > 0:
        #     calc_fscore(max_epoch=epoch_num)

        # avg_loss_hist = np.mean(loss_hist)
        # if avg_loss_hist < min_avg_loss_hist:
        #     min_avg_loss_hist = avg_loss_hist
        # else:
        #     learning_rate_base *= 0.1  # change learning rate
        #     optimizer = optim.Adam(
        #         model.parameters(), lr=learning_rate_base)


if __name__ == "__main__":
    main()
