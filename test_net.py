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
# from models.config import cfg
from tools.net_utils import get_current_lr
from collections import OrderedDict
from tools.net_utils import adjust_learning_rate
from lib.datasets.pascal_voc import get_target_size
from lib.datasets.pascal_voc import im_list_to_blob
from torchvision.ops import nms
# from models.config import cfg
import cv2
from models.resnet import resnet
from lib.datasets.pascal_voc import load_pascal_annotation

import xml.etree.ElementTree as ET

# added by Henson
from lib.config.config import Config, merge_config
from torch.backends import cudnn
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
from tqdm import tqdm

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])


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
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')
    parser.add_argument(
        '--base_config', default="lib/config/base_detection_config.py", help='config')
    parser.add_argument(
        '--config', default="lib/config/det_edge_config.py", help='config')
    parser.add_argument(
        '--max_epoch', default=1000, type=int, help='config')

    args = parser.parse_args()

    return args


def prepareTestData(target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE):
    batch_size = 1
    width, height, channles = get_target_size(
        target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE)
    im_scales = np.zeros((batch_size, 4), dtype=np.float32)
    gt_boxes = np.zeros((batch_size, 1, 5), dtype=np.float32)
    im_blobs = np.zeros(
        (batch_size, channles, height, width), dtype=np.float32)
    im_scale_x = float(width) / float(im.shape[1])
    im_scale_y = float(height) / float(im.shape[0])
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
    im_scale = np.array(
        [np.hstack((height, width, im_scale_x, im_scale_y))], dtype=np.float32)
    im = im.astype(np.float32) / 255
    im = (im - PIXEL_MEANS) / PIXEL_STDS
    im_blob = im_list_to_blob(im)
    im_blobs[0, :, :, :] = im_blob
    im_scales[0, :] = im_scale
    return [gt_boxes, im_blobs, im_scales]


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = ((ws >= min_size) & (hs >= min_size))
    return keep


def bbox_transform_inv(boxes, deltas, batch_size, std, mean):
    # print(boxes)
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4] * std[0] + mean[0]
    dy = deltas[:, :, 1::4] * std[1] + mean[1]
    dw = deltas[:, :, 2::4] * std[2] + mean[2]
    dh = deltas[:, :, 3::4] * std[3] + mean[3]
    # print(dx, dy, dw, dh)

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


def im_detect(data, model, batch_size, std, mean, cfg, nms_thresh):
    gt_tensor = torch.autograd.Variable(torch.from_numpy(data[0]))
    im_blobs_tensor = torch.autograd.Variable(torch.from_numpy(data[1]))
    im_info_tensor = torch.autograd.Variable(torch.from_numpy(data[2]))
    # print(im_info_tensor)
    results = []
    with torch.no_grad():
        rois, cls_prob, bbox_pred = model(im_blobs_tensor.cuda(),
                                          im_info_tensor.cuda(),
                                          gt_tensor.cuda())
        # print(rois[:,:,1:5])
        pred_boxes = bbox_transform_inv(
            rois[:, :, 1:5], bbox_pred, batch_size, std, mean)
        pred_boxes = clip_boxes(pred_boxes, im_info_tensor.data, 1)
        scores = cls_prob

        classes = len(cfg.class_list)
        for index in range(1, classes):
            cls_scores = scores[0, :, index]

            if not cfg.confidence_per_cls:
                raise RuntimeError("confidence_per_cls is not exist!!!")
            else:
                thresh = 0.5 #cfg.confidence_per_cls[cfg.class_list[index]][0]
                
            scores_over_thresh = (cls_scores > thresh)
            cls_keep = cls_scores[scores_over_thresh]
            bboxes_keep = pred_boxes[0,
                                     scores_over_thresh, index * 4: (index + 1) * 4]

            # filter_keep = _filter_boxes(bboxes_keep, 8)
            # cls_keep = cls_keep[filter_keep]
            # bboxes_keep = bboxes_keep[filter_keep, :]

            keep_idx_i = nms(bboxes_keep, cls_keep, nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)
            bboxes_keep = bboxes_keep[keep_idx_i, :]
            cls_keep = cls_keep[keep_idx_i]
            bboxes_keep[:, 0] /= im_info_tensor[0, 2]
            bboxes_keep[:, 1] /= im_info_tensor[0, 3]
            bboxes_keep[:, 2] /= im_info_tensor[0, 2]
            bboxes_keep[:, 3] /= im_info_tensor[0, 3]
            if bboxes_keep.size(0) > 0:
                result = np.zeros(
                    (bboxes_keep.size(0), 6), dtype=np.float32)
                result[:, 0:4] = bboxes_keep.cpu()
                result[:, 4] = cls_keep.cpu()
                result[:, 5] = index
                results.append(result)

            # threshold_list[key][0].append(result)

    return results


def get_all_annots(xml_path):
    xmls = glob(xml_path + '/*.xml')
    annots = []
    for xml in xmls:
        xml_context = load_pascal_annotation(xml)
        annot = np.zeros((xml_context['boxes'].shape[0], 5))
        annot[:, 0:4] = xml_context['boxes']
        annot[:, 4:5] = xml_context['gt_classes'].reshape(
            (xml_context['gt_classes'].shape[0], 1))
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
                dict_[str(label)] = np.row_stack(
                    (dict_[str(label)], np.array([x1, x2, y1, y2])))
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


def calc_iou(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[2] - b[0]) * (b[3] - b[1])

    iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])
    ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[2] - a[0]) * (a[3] - a[1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def det_init_dict(class_list, operate=[]):
    dicts = defaultdict(list)
    for cls_name in class_list:
        if cls_name != '__background__':
            dicts[cls_name].append(operate)

    return dicts


def det_results2dict(results, class_list, dicts=None):
    dicts = defaultdict(list)
    for result in results:
        for res in result:
            cls_name = class_list[int(res[-1])]
            if cls_name != '__background__':
                dicts[cls_name].append(res[0:5])

    return dicts


def xml2dict(rename_class_list, xml_name, class_list, dicts=None):
    dicts = defaultdict(list)
    objects = ET.parse(xml_name).findall("object")
    for object in objects:
        class_name = object.find('name').text.strip()
        class_name = class_name.upper()

        for key in rename_class_list.keys():
            if class_name in rename_class_list[key]:
                class_name = key
                break

        # vertical
        # if rename_class_list == "vertical":
        #     if class_name in ['DJ', 'CD', 'CJ', 'KT']:
        #         class_name = 'DJ'
        #     if class_name in ['JG', 'KZ', 'LS', 'SH', 'RK', 'HD']:
        #         class_name = 'JG'
        #     if class_name == 'KP1':
        #         class_name = 'KP'

        # horizon
        # if rename_class_list == "horizon":
        #     if class_name in ['KT', 'KZ', 'YJ', 'JJ', 'DJ', 'KQ', 'JG', 'MK']:
        #         class_name = 'KT'
        #     if class_name == 'KP1':
        #         class_name = 'KP'

        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)

        if '__background__' == class_name:
            continue
        obj = [xmin, ymin, xmax, ymax, 1]
        dicts[class_name].append(obj)

    return dicts


def show_matplot_image(x_axis, data, image_path):
    fig = plt.figure()
    plt.cla()
    plt.scatter(x_axis, data, marker='o', color='r', s=15)
    fig.savefig(image_path)


def show_matplot_image_3d(x_axis, data, data1=None, image_path=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x_axis, data, data1, marker='o', color='r', label='PR curve')
    ax.legend()
    fig.savefig(image_path)


def main(args, model_epoch=None):
    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg.model_cfg, det_cfg.model_cfg)

    if model_epoch is not None:
        cfg.model = os.path.join(
            cfg.TRAIN.save_dir, cfg.TRAIN.model_name + model_epoch)
        print(cfg.model)
        if not os.path.exists(cfg.model):
            return 0, 0, 0, 0, 0

    gpus = ",".join('%s' % id for id in cfg.TEST.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    thresh = cfg.TEST.thresh  # 0.8
    nms_thresh = cfg.TEST.nms_thresh  # 0.35
    iou_thresh = cfg.TEST.iou_thresh

    classes = len(cfg.class_list)
    if 'lite' in cfg.MODEL.BACKBONE:
        model = lite_faster_rcnn(cfg, classes)
    elif 'pva' in cfg.MODEL.BACKBONE:
        model = pva_net(classes)
    elif 'resnet' in cfg.MODEL.BACKBONE:
        model = resnet(classes, num_layers=101)

    model.create_architecture(cfg)

    checkpoint = torch.load(cfg.model)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")
    model.eval()

    imgs = glob(os.path.join(cfg.TEST.img_path, "*.xml"))
    # print(cfg.TEST.img_path)
    batch_size = 1
    std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
    mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
    std = torch.from_numpy(std).cuda()
    mean = torch.from_numpy(mean).cuda()
    all_detections = []

    gt_box_num_datasets = dict.fromkeys(cfg.class_list, 0)
    det_box_num_datasets = dict.fromkeys(cfg.class_list, 0)
    undetected_sample = []
    undetected_sample_size = defaultdict(list)
    precsion_dicts = det_init_dict(cfg.class_list, 0)
    recall_dicts = det_init_dict(cfg.class_list, 0)
    TP = det_init_dict(cfg.class_list, 0)
    FP = det_init_dict(cfg.class_list, 0)
    FN = det_init_dict(cfg.class_list, 0)
    total_image_nums = len(imgs)
    for ix, img_name in enumerate(tqdm(imgs)):
        img_name = img_name.replace(".xml", ".jpg")
        # print(ix, img_name)
        im = cv2.imread(img_name)
        if im is None:
            continue
        import time
        # start_t = time.clock()
        data = prepareTestData(cfg.TEST.TEST_SCALE, im,
                               cfg.TEST.SCALE_MULTIPLE_OF, cfg.TEST.MAX_SIZE)
        start_t = time.time()
        results = im_detect(data, model, batch_size, std,
                            mean, cfg, nms_thresh)
        end_t = time.time()
        # print(ix, "/", total_image_nums, img_name, ' time consume is: ',
            #   end_t - start_t)

        if 1:  # len(results) > 0:
            xml_name = img_name.replace(".jpg", ".xml")
            xml_dicts = xml2dict(cfg.rename_class_list,
                                 xml_name, cfg.class_list)
            det_dicts = det_results2dict(results, cfg.class_list)

            small_object_size = cfg.TEST.small_object_size
            for key in cfg.class_list:
                if key == '__background__':
                    continue

                ignore_gt_box_num_per_cls = dict.fromkeys(cfg.class_list, 0)
                ignore_det_box_num_per_cls = dict.fromkeys(cfg.class_list, 0)
                is_match_gt = np.zeros(len(xml_dicts[key]))
                is_match_det = np.zeros(len(det_dicts[key]))

                for gt_id, gt_box in enumerate(xml_dicts[key]):
                    gt_s0 = gt_box[3] - gt_box[1]
                    gt_s1 = gt_box[2] - gt_box[0]

                    if gt_s0 < small_object_size or gt_s1 < small_object_size:
                        ignore_gt_box_num_per_cls[key] += 1.0
                        is_match_gt[gt_id] = -1.0

                gt_box_num = len(xml_dicts[key]) - ignore_gt_box_num_per_cls[key]
                gt_box_num_datasets[key] += gt_box_num
                # print(xml_name, key, gt_box_num, gt_box_num_datasets[key])

                for det_id, det_box in enumerate(det_dicts[key]):
                    det_s0 = det_box[3] - det_box[1]
                    det_s1 = det_box[2] - det_box[0]
                    if det_s0 < small_object_size or det_s1 < small_object_size:
                        ignore_det_box_num_per_cls[key] += 1.0
                        is_match_det[det_id] = -1.0
                        continue
                    
                    max_iou, max_iou_id = 0, 0
                    for gt_id, gt_box in enumerate(xml_dicts[key]):
                        if abs(is_match_gt[gt_id]) > 0:
                            continue

                        iou = calc_iou(det_box[0:-1], gt_box[0:-1])
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_id = gt_id

                        if gt_id == len(xml_dicts[key]) - 1:
                            if max_iou > iou_thresh:
                                is_match_gt[max_iou_id] = 1.0
                                is_match_det[det_id] = 1.0

                det_box_num = len(det_dicts[key]) - ignore_det_box_num_per_cls[key]
                det_box_num_datasets[key] += det_box_num

                for object_id in range(len(is_match_gt)):
                    if is_match_gt[object_id] == 0:
                        gt_box = xml_dicts[key][object_id]
                        gt_s0 = gt_box[3] - gt_box[1]
                        gt_s1 = gt_box[2] - gt_box[0]
                        undetected_sample_size[key].append(
                            (os.path.basename(xml_name), object_id, gt_s0, gt_s1))
                        # print(img_name, len(
                        #     xml_dicts[key]) - ignore_gt_box_num_per_cls[key])
                        # save_dir = cfg.TEST.save_dir
                        # name = os.path.basename(img_name)
                        # shutil.copyfile(img_name, save_dir +
                        #                 'undetected_sample/' + name)
                        # shutil.copyfile(img_name.replace(".jpg", ".xml"), save_dir + 'undetected_sample/' +
                        #                 name.replace(".jpg", ".xml"))

                tp = np.sum(is_match_det)
                fp = np.sum(abs(is_match_det - 1))
                fn = np.sum(abs(is_match_gt - 1))
                TP[key][0] += tp
                FP[key][0] += fp
                FN[key][0] += fn
                
                recall_cls = np.sum(is_match_gt)
                recall_dicts[key].append(recall_cls)
                if gt_box_num > 0 and recall_cls / gt_box_num < 1.0 and not (xml_name in undetected_sample):
                    undetected_sample.append(xml_name)

                precision_cls = np.sum(is_match_gt)
                precsion_dicts[key].append(precision_cls)

        # if ix > 100:
        #     break

    avg_precision = 0.0
    avg_recall = 0.0
    avg_fscore = 0.0
    cls_num = 0.0
    for key in cfg.defect_class_list:
        # print("recall: ", np.sum(recall_dicts[key]),
        #       "gt_box_num_datasets: ", gt_box_num_datasets[key],
        #       "det_box_num_datasets: ", det_box_num_datasets[key])

        # recall_per_cls, precision_per_cls = 0, 0
        # if gt_box_num_datasets[key] > 0:
        #     cls_num = cls_num + 1
        #     recall_per_cls = np.sum(
        #         recall_dicts[key]) / gt_box_num_datasets[key]

        # if det_box_num_datasets[key] > 0:
        #     precision_per_cls = np.sum(
        #         precsion_dicts[key]) / det_box_num_datasets[key]

        recall_per_cls = TP[key][0] / (TP[key][0] + FN[key][0])
        precision_per_cls = TP[key][0] / (TP[key][0] + FP[key][0])
        fscore_per_cls = 2 * recall_per_cls * precision_per_cls / (recall_per_cls + precision_per_cls)
        
        cls_num = cls_num + 1

        if gt_box_num_datasets[key] > 0:
            print("class_name: ", key,
                  "recall_per_cls: ", recall_per_cls,
                  "precision_per_cls: ", precision_per_cls,
                  "fscore_per_cls: ", fscore_per_cls,
                  "gt_box_num_datasets: ", gt_box_num_datasets[key],
                  "det_box_num_datasets: ", det_box_num_datasets[key])

            avg_recall += recall_per_cls
            avg_precision += precision_per_cls
            avg_fscore += fscore_per_cls

    avg_recall = avg_recall / cls_num
    avg_precision = avg_precision / cls_num
    avg_fscore = avg_fscore / cls_num
    undetected_ratio = len(undetected_sample) / len(imgs)

    print("avg_recall: ", avg_recall)
    print("avg_precision: ", avg_precision)
    print("avg_fscore: ", avg_fscore)
    print("undetected_ratio: ", undetected_ratio)
    # print("undetected_sample: ", undetected_sample)

    with open("undetected_sample_size.txt", 'w') as f:
        for key in undetected_sample_size.keys():
            for undetected_gt_size in undetected_sample_size[key]:
                # print("key: ", key, undetected_gt_size)
                f.write("key: " + key + " " + str(undetected_gt_size) + "\n")

                if key in cfg.defect_class_list:
                    save_dir = cfg.TEST.save_dir
                    name = undetected_gt_size[0]

                    src_file = save_dir + name.replace(".xml", ".jpg")
                    dst_file = save_dir + 'undetected_sample/' + \
                        name.replace(".xml", ".jpg")
                    if not os.path.exists(dst_file) and os.path.exists(src_file):
                        shutil.copyfile(src_file, dst_file)

                    src_file = save_dir + name.replace(".xml", ".bmp")
                    dst_file = save_dir + 'undetected_sample/' + name.replace(".xml", ".bmp")
                    if not os.path.exists(dst_file) and os.path.exists(src_file):
                        shutil.copyfile(src_file, dst_file)

        f.close()

    # with open("undetected_sample.txt", 'w') as f:
    #     for key in undetected_sample_size.keys():
    #         for undetected_gt_size in undetected_sample_size[key]:
    #             f.write(str(undetected_gt_size[0]) + "\n")
    #     f.close()

    return avg_recall, avg_precision, avg_fscore, undetected_ratio, 1


def calc_fscore():
    args = parse_args()

    undetected_ratios = []
    avg_recalls = []
    avg_precisions = []
    fscores = []
    x_axis = []

    max_epoch = 110  # args.max_epoch
    epoch_interval = 1
    test_epoch_num = 1
    for epoch in range(max_epoch, max(max_epoch - test_epoch_num, 0), -epoch_interval):
        if epoch == 0:
            break

        if epoch == -1:
            continue

        model_epoch = '%04d.ckpt' % epoch
        # print(model_epoch)
        # continue

        avg_recall, avg_precision, fscore, undetected_ratio, flag = main(
            args, model_epoch)
        if flag == 0:
            continue

        x_axis.append(epoch)
        undetected_ratios.append(undetected_ratio)
        avg_recalls.append(avg_recall)
        avg_precisions.append(avg_precision)
        fscores.append(fscore)

        show_matplot_image(x_axis, undetected_ratios, "undetected_ratios.jpg")
        show_matplot_image(x_axis, avg_recalls, "recall.jpg")
        show_matplot_image(x_axis, avg_precisions, "precision.jpg")
        show_matplot_image(x_axis, fscores, "fscore.jpg")
        show_matplot_image(avg_precisions,
                           avg_recalls, "PR_curve.jpg")


def test(args, model_epoch=None):
    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg.model_cfg, det_cfg.model_cfg)

    if model_epoch is not None:
        cfg.model = os.path.join(
            cfg.TRAIN.save_dir, cfg.TRAIN.model_name + model_epoch)
        print(cfg.model)
        if not os.path.exists(cfg.model):
            return 0, 0, 0, 0, 0

    gpus = ",".join('%s' % id for id in cfg.TEST.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    thresh = cfg.TEST.thresh  # 0.8
    nms_thresh = cfg.TEST.nms_thresh  # 0.35
    iou_thresh = cfg.TEST.iou_thresh

    classes = len(cfg.class_list)
    if 'lite' in cfg.MODEL.BACKBONE:
        model = lite_faster_rcnn(cfg, classes)
    elif 'pva' in cfg.MODEL.BACKBONE:
        model = pva_net(classes)
    elif 'resnet' in cfg.MODEL.BACKBONE:
        model = resnet(classes, num_layers=101)

    model.create_architecture(cfg)

    checkpoint = torch.load(cfg.model)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")
    model.eval()

    imgs = glob(os.path.join(cfg.TEST.img_path, "*.xml"))
    # print(cfg.TEST.img_path)
    batch_size = 1
    std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
    mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
    std = torch.from_numpy(std).cuda()
    mean = torch.from_numpy(mean).cuda()

    total_image_nums = len(imgs)
    for ix, img_name in enumerate(imgs):
        img_name = img_name.replace(".xml", ".jpg")
        # print(ix, img_name)
        im = cv2.imread(img_name)
        if im is None:
            continue
        import time
        # start_t = time.clock()
        data = prepareTestData(cfg.TEST.TEST_SCALE, im,
                               cfg.TEST.SCALE_MULTIPLE_OF, cfg.TEST.MAX_SIZE)
        start_t = time.time()
        results = im_detect(data, model, batch_size, std,
                            mean, classes, thresh, nms_thresh)
        end_t = time.time()
        # print(ix, "/", total_image_nums, img_name, ' time consume is: ',
        #       end_t - start_t, len(results))
        print(ix, "/", total_image_nums, img_name, ' time consume is: ',
              end_t - start_t)

        xml_name = img_name.replace(".jpg", ".xml")
        xml_dicts = xml2dict(cfg.rename_class_list,
                             xml_name, cfg.class_list)
        det_dicts = det_results2dict(results, cfg.class_list)

        txt_name = os.path.basename(xml_name).replace('.xml', '.txt')

        # os.remove("mAP/input/ground-truth/")
        if not os.path.exists("mAP/input/ground-truth/"):
            os.makedirs("mAP/input/ground-truth/")

        # os.remove("mAP/input/detection-results/")
        if not os.path.exists("mAP/input/detection-results/"):
            os.makedirs("mAP/input/detection-results/")

        # GT
        with open("mAP/input/ground-truth/" + txt_name, 'w') as f:
            for key in cfg.class_list:
                if key == '__background__':
                    continue

                if key == 'FC':
                    break

                for gt_id, gt_box in enumerate(xml_dicts[key]):
                    f.write(key + " "
                            + str(gt_box[0]) + " "
                            + str(gt_box[1]) + " "
                            + str(gt_box[2]) + " "
                            + str(gt_box[3]) + "\n")
            f.close()

        # DET results
        with open("mAP/input/detection-results/" + txt_name, 'w') as f:
            for key in cfg.class_list:
                if key == '__background__':
                    continue

                if key == 'FC':
                    break

                for det_id, det_box in enumerate(det_dicts[key]):
                    f.write(key + " "
                            + str(det_box[-1]) + " "
                            + str(det_box[0]) + " "
                            + str(det_box[1]) + " "
                            + str(det_box[2]) + " "
                            + str(det_box[3]) + "\n")
            f.close()


def calc_map():
    args = parse_args()

    epoch = 872
    model_epoch = '%04d.ckpt' % epoch

    test(args, model_epoch)


if __name__ == "__main__":
    # calc_map()
    calc_fscore()
