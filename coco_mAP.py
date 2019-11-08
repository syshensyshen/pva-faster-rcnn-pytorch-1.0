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


def im_detect(data, model, batch_size, std, mean, cfg, nms_thresh=0.25):
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

            thresh = 0.0  # round(1 / classes, 4)
            # if not cfg.confidence_per_cls:
            #     thresh = 1 / classes
            # else:
            #     thresh = cfg.confidence_per_cls[cfg.class_list[index]]

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

    return results


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


def det_init_dict(class_list, dicts=None, operate=[]):
    if dicts is None:
        dicts = defaultdict(list)

    for cls_name in class_list:
        if cls_name != '__background__':
            dicts[cls_name].append(operate)

    return dicts


def det_results2dict(xml_name, results, class_list, dicts=None):
    for result in results:
        for id, res in enumerate(result):
            cls_name = class_list[int(res[-1])]
            if cls_name != '__background__' and res[-2] > 1 / len(class_list):
                dicts[(xml_name, cls_name)].append(res[0:5])

    return dicts


def filter_detbox(dicts, dicts_original, confidence):
    for key in dicts_original.keys():
        for det_box in dicts_original[key]:
            if det_box[-1] > confidence:
                dicts[key].append(det_box)

    return dicts


def xml2dict(rename_class_list, xml_name, class_list, dicts=None):
    objects = ET.parse(xml_name).findall("object")
    for id, object in enumerate(objects):
        class_name = object.find('name').text.strip()
        class_name = class_name.upper()

        for key in rename_class_list.keys():
            if class_name in rename_class_list[key]:
                class_name = key
                break

        if not class_name in class_list:
            continue

        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)

        if '__background__' == class_name:
            continue

        obj = [xmin, ymin, xmax, ymax, 1]
        dicts[(xml_name, class_name)].append(obj)

    return dicts


def detections(cfg, model_epoch):
    if model_epoch is not None:
        cfg.model = os.path.join(
            cfg.TRAIN.save_dir, cfg.TRAIN.model_name + model_epoch)
        print(cfg.model)
        if not os.path.exists(cfg.model):
            raise RuntimeError("model is not exist!!!")

    gpus = ",".join('%s' % id for id in cfg.TEST.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    thresh = cfg.TEST.thresh  # 0.8
    nms_thresh = cfg.TEST.nms_thresh  # 0.35

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

    xml_names = []
    xml_dicts = defaultdict(list)
    det_dicts = defaultdict(list)
    total_image_nums = len(imgs)
    for ix, img_name in enumerate(tqdm(imgs)):
        img_name = img_name.replace(".xml", cfg.image_postfix)
        # print(ix, img_name)
        im = cv2.imread(img_name)
        if im is None:
            continue

        import time

        data = prepareTestData(cfg.TEST.TEST_SCALE, im,
                               cfg.TEST.SCALE_MULTIPLE_OF, cfg.TEST.MAX_SIZE)
        start_t = time.time()
        results = im_detect(data, model, batch_size, std,
                            mean, cfg, nms_thresh)
        end_t = time.time()

        # print(ix, "/", total_image_nums, img_name, ' time consume is: ',
        #       end_t - start_t)

        xml_name = img_name.replace(cfg.image_postfix, ".xml")
        xml_names.append(xml_name)

        xml_dicts = xml2dict(cfg.rename_class_list,
                             xml_name, cfg.class_list, xml_dicts)
        det_dicts = det_results2dict(
            xml_name, results, cfg.class_list, det_dicts)

        # if ix > 100:
        #     break

    return xml_dicts, det_dicts, xml_names


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


def coco_map(xml_dicts, det_dicts_original, xml_names, class_list, defect_class_list=None, iou_thresh=0.5, small_object_size=0):
    map_esp = 10e-8
    total_cls_num = len(class_list)
    # confidence_list = [round(conf / total_cls_num, 4)
    #                    for conf in range(0, total_cls_num + 1)]
    confidence_list = [round(conf / 100.0, 4)
                       for conf in range(0, 100 + 1, 1)]
    precision_dicts = defaultdict(list)
    recall_dicts = defaultdict(list)
    fscore_dicts = defaultdict(list)

    for confidence in confidence_list:
        det_dicts = defaultdict(list)
        det_dicts = filter_detbox(
            det_dicts, det_dicts_original, confidence)

        TP_per_cls = dict.fromkeys(class_list, 0)
        FP_per_cls = dict.fromkeys(class_list, 0)
        FN_per_cls = dict.fromkeys(class_list, 0)
        # gt_box_num_datasets = dict.fromkeys(class_list, 0)
        # det_box_num_datasets = dict.fromkeys(class_list, 0)

        for xml_name in xml_names:
            for key in class_list:
                if key == '__background__':
                    continue

                ignore_gt_box_num_per_cls = dict.fromkeys(class_list, 0)
                ignore_det_box_num_per_cls = dict.fromkeys(class_list, 0)
                is_match_gt = np.zeros(len(xml_dicts[(xml_name, key)]))
                is_match_det = np.zeros(len(det_dicts[(xml_name, key)]))

                for gt_id, gt_box in enumerate(xml_dicts[(xml_name, key)]):
                    gt_s0 = gt_box[3] - gt_box[1]
                    gt_s1 = gt_box[2] - gt_box[0]

                    if gt_s0 < small_object_size or gt_s1 < small_object_size:
                        ignore_gt_box_num_per_cls[key] += 1.0
                        is_match_gt[gt_id] = -1.0

                # gt_box_num_datasets[key] += len(xml_dicts[(xml_name, key)]) - \
                #     ignore_gt_box_num_per_cls[key]
                # print("xml_name: ", os.path.basename(xml_name),
                #       "key: ", key, "gt_box_num_datasets: ", gt_box_num_datasets[key])

                for det_id, det_box in enumerate(det_dicts[(xml_name, key)]):
                    det_s0 = det_box[3] - det_box[1]
                    det_s1 = det_box[2] - det_box[0]
                    if det_s0 < small_object_size or det_s1 < small_object_size:
                        ignore_det_box_num_per_cls[key] += 1.0
                        is_match_det[det_id] = -1.0
                        continue

                    max_iou, max_iou_id = 0, 0
                    for gt_id, gt_box in enumerate(xml_dicts[(xml_name, key)]):
                        if abs(is_match_gt[gt_id]) > 0:
                            continue

                        iou = calc_iou(det_box[0:-1], gt_box[0:-1])
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_id = gt_id

                    if max_iou > iou_thresh:
                        is_match_gt[max_iou_id] = 1.0
                        is_match_det[det_id] = 1.0

                # det_box_num_datasets[key] += len(
                #     det_dicts[(xml_name, key)]) - ignore_det_box_num_per_cls[key]
                # print("xml_name: ", os.path.basename(xml_name),
                #       "key: ", key, "det_box_num_datasets: ", det_box_num_datasets[key])

                tp = np.sum(is_match_det)
                fp = np.sum(abs(is_match_det - 1))
                fn = np.sum(abs(is_match_gt - 1))
                # print("tp: ", tp, "fp: ", fp, "fn: ", fn)

                TP_per_cls[key] += tp
                FP_per_cls[key] += fp
                FN_per_cls[key] += fn

        for key in class_list:
            if key == "__background__":
                continue

            denominator = (TP_per_cls[key] + FP_per_cls[key]) + map_esp
            precision = round(TP_per_cls[key] / denominator, 4)
            precision_dicts[(key, str(confidence))].append(precision)

            denominator = (TP_per_cls[key] + FN_per_cls[key]) + map_esp
            recall = round(TP_per_cls[key] / denominator, 4)
            recall_dicts[(key, str(confidence))].append(recall)

            denominator = (precision + recall) + map_esp
            fscore = round(2 * precision * recall / denominator, 4)
            fscore_dicts[(key, str(confidence))].append(fscore)

        #     print(" class_name: ", key,
        #           " confidence: ", confidence,
        #           " fscore: ", fscore_dicts[(key, str(confidence))],
        #           " recall: ", recall,
        #           " precision: ", precision)
        # print("\n")

    mAP = det_init_dict(class_list, None, round(0.0, 4))
    best_fscore = det_init_dict(class_list, None, round(0.0, 4))
    best_confidence = det_init_dict(
        class_list, None, round(1 / len(class_list), 4))
    for key in class_list:
        if key == "__background__":
            continue

        axis_c, axis_x, axis_y = [], [], []
        for i in range(len(confidence_list) - 1):
            axis_c.append(confidence_list[i])
            axis_x.append(recall_dicts[(
                key, str(confidence_list[i]))][0])
            axis_y.append(precision_dicts[(
                key, str(confidence_list[i]))][0])

        axis_xs, axis_ys = [], []
        axis_xs.append(axis_x[0])
        axis_ys.append(axis_y[0])
        for i in range(1, len(axis_x)):
            if abs(axis_x[i] - axis_x[i - 1]) > 1.0e-5:
                axis_xs.append(axis_x[i])
                axis_ys.append(axis_y[i])

        show_matplot_image(axis_xs, axis_ys, key + ".jpg")

        for i in range(1, len(axis_xs)):
            delta_recall = abs(axis_xs[i - 1] - axis_xs[i])
            delta_precision = min(axis_ys[i], axis_ys[i - 1])

            mAP[key][0] += delta_recall * delta_precision

        for confidence in confidence_list:
            fscore = fscore_dicts[(key, str(confidence))]
            if len(fscore) == 0:
                continue
            #     raise RuntimeError("this class_name is not exist!!!")

            if fscore[0] >= best_fscore[key][0]:
                best_fscore[key][0] = fscore[0]
                best_confidence[key][0] = confidence

        print(" key: ", key,
              " best_confidence: ", best_confidence[key],
              " best_fscore: ", best_fscore[key],
              " best_recall: ", recall_dicts[(
                  key, str(best_confidence[key][0]))],
              " best_precision: ", precision_dicts[(
                  key, str(best_confidence[key][0]))])

    avg_fscore_defect = 0.0
    for key in defect_class_list:
        avg_fscore_defect += best_fscore[key][0]
    avg_fscore_defect /= len(defect_class_list)

    avg_fscore = 0.0
    for key in class_list:
        if key == "__background__" or key in defect_class_list:
            continue
        avg_fscore += best_fscore[key][0]
    avg_fscore /= (len(class_list) - len(defect_class_list) - 1)

    print(" avg_fscore_defect: ", avg_fscore_defect)
    print(" avg_fscore: ", avg_fscore)
    print(" best_confidence: ", best_confidence)

    return avg_fscore, best_confidence, mAP


def main(args, model_epoch=None):
    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg.model_cfg, det_cfg.model_cfg)
    print(args.config)

    iou_thresh = cfg.TEST.iou_thresh
    small_object_size = cfg.TEST.small_object_size
    xml_dicts, det_dicts_original, xml_names = detections(cfg, model_epoch)

    avg_fscore, best_confidence, mAP = coco_map(
        xml_dicts, det_dicts_original, xml_names, cfg.class_list, cfg.defect_class_list, iou_thresh, small_object_size)

    return avg_fscore, best_confidence, mAP


def calc_fscore(max_epoch):
    args = parse_args()

    # max_epoch = 330  # args.max_epoch
    epoch_interval = 1
    test_epoch_num = 1
    for epoch in range(max_epoch, max(max_epoch - test_epoch_num, 0), -epoch_interval):
        if epoch == 0:
            break

        if epoch == -1:
            continue

        model_epoch = '%04d.ckpt' % epoch
        avg_fscore, best_confidence, mAP = main(args, model_epoch)


if __name__ == "__main__":
    calc_fscore(max_epoch=434)
