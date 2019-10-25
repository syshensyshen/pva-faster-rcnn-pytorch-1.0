'''
author: syshen
date: 2019/01/22-01/23
'''
import numpy as np
import argparse
import torch
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os

from datasets.dataloader import CocoDataset, CSVDataset, XML_VOCDataset
from datasets.dataloader import collater, Resizer, AspectRatioBasedSampler
from datasets.dataloader import Augmenter, Normalizer
from torch.utils.data import DataLoader

from models.detector import detector

from config.config import Config, merge_config


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')
    parser.add_argument(
        '--base_config', default="config/base_detection_config.py", help='config')
    parser.add_argument(
        '--config', default="config/fpn_syshen.py", help='config')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg, det_cfg)

    gpus = ",".join('%s' % id for id in cfg.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpu_nums = len(gpus.split(','))

    if cfg.dataset == 'coco':

        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Augmenter()]))
        dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Augmenter()]))

    elif cfg.dataset == 'csv':

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

    elif cfg.dataset == 'xml':
        if cfg.train_path is None:
            raise ValueError(
                'Must provide --voc_train when training on PASCAL VOC,')
        dataset_train = XML_VOCDataset(cfg, img_path=cfg.train_path,
                                       xml_path=cfg.train_path, class_list=cfg.class_list,
                                       transform=transforms.Compose([Augmenter()]))

        if not cfg.val_path:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = XML_VOCDataset(cfg, img_path=cfg.val_path, xml_path=cfg.val_path,
                                         class_list=cfg.class_list, transform=transforms.Compose([]))

    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=cfg.batch_size_per_gpu * gpu_nums, drop_last=False)
    dataloader_train = DataLoader(
        dataset_train, num_workers=cfg.num_works, collate_fn=collater, batch_sampler=sampler, pin_memory=False)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(
            dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val, pin_memory=False)

    print('Num training images: {}'.format(len(dataset_train)))
    print("Training models...")
    classes = len(cfg.class_list)

    model = detector(cfg)

    start_epoch = 0
    if cfg.resume_model:  # added by Henson
        checkpoint = torch.load(cfg.resume_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("loading pretrained model: ", cfg.resume_model)

    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True
        model = model.cuda()
        if gpu_nums > 1:
            model = nn.DataParallel(model)

    model.train()

    learning_rate_base = cfg.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, verbose=True, mode="max")
    loss_hist = collections.deque(maxlen=1024)
    # min_loss = 1.0
    min_avg_loss_hist = 110000.0
    for epoch_num in range(start_epoch, cfg.epochs):
        print("\n=> learning_rate: ", learning_rate_base,
              " min_avg_loss_hist: ", min_avg_loss_hist)

        # epoch_loss = []
        loss_hist.clear()
        # loss, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox = 0, 0, 0, 0, 0
        for iter_num, data in enumerate(dataloader_train):

            try:
                _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox = model(
                    data['img'].cuda(), data['im_info'].cuda(), data['annot'].cuda())

                loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + loss_cls.mean() + loss_bbox.mean()
                if bool(loss == 0):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # epoch_loss.append(float(loss))
                if gpu_nums > 1:
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

                loss_hist.append(float(loss))

                if iter_num % 20 == 0:
                    output_str = 'Epoch: {} | Iteration: {}/{} | loss: {:1.5f} | rpn bbox loss: {:1.5f} | rpn cls loss: {:1.5f} | bbox loss: {:1.5f} | cls loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, len(dataset_train)//(cfg.batch_size_per_gpu * gpu_nums), float(loss), float(rpn_loss_bbox), float(rpn_loss_cls), float(loss_bbox), float(loss_cls), np.mean(loss_hist))
                    print(output_str)
                    logs = open('./logs/train.log', 'a+')
                    logs.write(output_str + '\n')
                    logs.close()

                # del rpn_loss_cls, rpn_loss_bbox, loss_bbox, loss_cls
            except Exception as e:
                print('Epoch: {} | Iteration: {}/{} | Exception: {}'.format(epoch_num, iter_num, len(
                    dataset_train)//(cfg.batch_size_per_gpu * gpu_nums), e))
                continue

        if epoch_num % 80 == 0 and epoch_num > 0:
            learning_rate_base /= 10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)

        if epoch_num % cfg.save_model_interval == 0 and epoch_num != 0:
            if gpu_nums > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_model_path = os.path.join(
                cfg.save_dir, cfg.model_name + '%04d.ckpt' % epoch_num)
            print("save model: ", save_model_path)
            torch.save({
                'epoch': epoch_num,
                'save_dir': cfg.save_dir,
                'state_dict': state_dict}, save_model_path)


if __name__ == "__main__":
    main()
