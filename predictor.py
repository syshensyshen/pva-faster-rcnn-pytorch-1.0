'''
author: syshen
date: 2019/01/28-01/30
'''
import numpy as np
import argparse
from glob import glob
import torch
from lib.rpn.bbox_transform import clip_boxes
import os
from models.lite import lite_faster_rcnn
from models.pvanet import pva_net
# from models.config import cfg
from datasets import get_target_size
from datasets import im_list_to_blob
# from lib.roi_layers.nms import nms
from torchvision.ops import nms
# from models.config import cfg
import cv2
from models.resnet import resnet
from tools.pascal_voc import PascalVocWriter

# added by Henson
import xml.etree.ElementTree as ET
from config.config import Config, merge_config
from collections import defaultdict

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
        '--config', default="lib/config/det_whiteshow_config.py", help='config')

    args = parser.parse_args()

    return args


def prepareTestData(target_size, im, SCALE_MULTIPLE_OF, MAX_SIZE):
    batch_size = 1
    width, height, channles = get_target_size(target_size, im, 32, MAX_SIZE)
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

    if 1:
        pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
        pred_w = torch.exp(dw) * widths.unsqueeze(2)
        pred_h = torch.exp(dh) * heights.unsqueeze(2)
    else:
        pred_ctr_x = ctr_x.unsqueeze(2)
        pred_ctr_y = ctr_y.unsqueeze(2)
        pred_w = widths.unsqueeze(2)
        pred_h = heights.unsqueeze(2)

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
    # print(im_blobs_tensor.shape)
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
        # results = []
        # print(rois.shape, scores.shape, rois.shape, bbox_pred.shape, classes)
        classes = len(cfg.class_list)
        for index in range(1, classes):
            cls_scores = scores[0, :, index]

            if not cfg.confidence_per_cls:
                raise RuntimeError("confidence_per_cls is not exist!!!")
            else:
                # cfg.confidence_per_cls[cfg.class_list[index]][0]
                thresh = cfg.confidence_per_cls[cfg.class_list[index]][0]

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
                result = np.zeros((bboxes_keep.size(0), 6), dtype=np.float32)
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

        if not class_name in class_list:
            continue

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


def save_image_gt(xml_image, xml_image_name, color_list, save_dir,
                  rename_class_list, xml_name, class_list):
    objects = ET.parse(xml_name).findall("object")
    for object in objects:
        class_name = object.find('name').text.strip()
        class_name = class_name.upper()

        for key in rename_class_list.keys():
            if class_name in rename_class_list[key]:
                class_name = key
                break

        if not class_name in class_list:
            continue

        x1 = int(object.find('bndbox/xmin').text)
        y1 = int(object.find('bndbox/ymin').text)
        x2 = int(object.find('bndbox/xmax').text)
        y2 = int(object.find('bndbox/ymax').text)

        g, b, r = color_list[class_name][0]
        w = x2 - x1
        h = y2 - y1

        cv2.rectangle(
            xml_image, (x1, y1), (x2, y2), (g, b, r), 1)
        cv2.putText(xml_image, class_name, (x1 + w//2, y1 + h//2),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (g, b, r), 1)  # label*50, label*100, label*100

    name = os.path.basename(xml_image_name)
    cv2.imwrite(save_dir + '/' + name, xml_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if not os.path.exists(save_dir + '/gt'):
        os.makedirs(save_dir + '/gt')
    cv2.imwrite(save_dir + '/gt/' + name, xml_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():
    args = parse_args()

    base_cfg = Config.fromfile(args.base_config)
    det_cfg = Config.fromfile(args.config)
    cfg = merge_config(base_cfg.model_cfg, det_cfg.model_cfg)

    gpus = ",".join('%s' % id for id in cfg.TEST.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    thresh = cfg.TEST.thresh
    nms_thresh = cfg.TEST.nms_thresh

    save_dir = cfg.TEST.save_dir
    classes = len(cfg.class_list)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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
    print(cfg.TEST.img_path)
    batch_size = 1
    std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
    mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
    std = torch.from_numpy(std).cuda()
    mean = torch.from_numpy(mean).cuda()

    color_list = defaultdict(list)
    for key in cfg.class_list:
        bgr = (np.random.randint(10, 255), np.random.randint(
            10, 255), np.random.randint(10, 255))
        color_list[key].append(bgr)

    total_image_nums = len(imgs)
    undetected_sample = []
    for ix, img_name in enumerate(imgs):
        save_flag = 0

        img_name = img_name.replace(".xml", cfg.image_postfix)
        # print(ix, img_name)
        im = cv2.imread(img_name)
        image_cpy = im.copy()
        xml_image = im.copy()
        gt_image = im.copy()

        if im is None:
            continue
        import time
        # start_t = time.clock()
        # if args.gama:
        #   im = adjust_gamma(im, 2.0)
        data = prepareTestData(cfg.TEST.TEST_SCALE, im,
                               cfg.TEST.SCALE_MULTIPLE_OF, cfg.TEST.MAX_SIZE)
        start_t = time.time()
        results = im_detect(data, model, batch_size, std,
                            mean, cfg, nms_thresh)
        end_t = time.time()
        print(ix, "/", total_image_nums, img_name,
              ' time consume is: ', end_t - start_t, len(results))

        # show undetected sample gt
        xml_name = img_name.replace(cfg.image_postfix, ".xml")
        if os.path.exists(xml_name):
            xml_dicts = xml2dict(cfg.rename_class_list,
                                 xml_name, cfg.class_list)
            det_dicts = det_results2dict(results, cfg.class_list)
            is_undetected = 0

            for key in cfg.class_list:
                if key == '__background__':
                    continue

                is_match_gt = np.zeros(len(xml_dicts[key]))
                is_match_det = np.zeros(len(det_dicts[key]))

                for det_id, det_box in enumerate(det_dicts[key]):
                    max_iou, max_iou_id = 0, 0
                    for gt_id, gt_box in enumerate(xml_dicts[key]):
                        if abs(is_match_gt[gt_id]) > 0:
                            continue

                        iou = calc_iou(det_box[0:-1], gt_box[0:-1])
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_id = gt_id

                    if max_iou > cfg.TEST.iou_thresh:
                        is_match_gt[max_iou_id] = 1.0
                        is_match_det[det_id] = 1.0

                for object_id in range(len(is_match_gt)):
                    if is_match_gt[object_id] == 0:
                        # is_undetected += 1
                        # # show_gt_image
                        # if is_undetected == 1:
                        #     xml_image = im.copy()
                        #     xml_image_name = img_name.replace(".jpg", ".bmp")
                        #     save_image_gt(xml_image, xml_image_name, color_list, save_dir,
                        #                   cfg.rename_class_list, xml_name, cfg.class_list)

                        gt_box = xml_dicts[key][object_id]
                        save_dir = cfg.TEST.save_dir
                        name = os.path.basename(xml_name).split(
                            ".xml")[0] + "_" + str(object_id) + ".bmp"

                        dst_file = save_dir + '/undetected_sample/' + key + "/"
                        if not os.path.exists(dst_file):
                            os.makedirs(dst_file)
                        dst_file += name

                        image_roi = image_cpy[
                            gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]]

                        if not os.path.exists(dst_file):
                            cv2.imwrite(dst_file, image_roi, [
                                int(cv2.IMWRITE_JPEG_QUALITY), 100])

                        g, b, r = color_list[key][0]
                        x1, x2 = gt_box[0], gt_box[2]
                        y1, y2 = gt_box[1], gt_box[3]
                        w = x2 - x1
                        h = y2 - y1
                        cv2.rectangle(
                            xml_image, (x1, y1), (x2, y2), (g, b, r), 1)
                        cv2.putText(xml_image, key, (x1 + w//2, y1 + h//2),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (g, b, r), 1)
                        if save_flag == 0 or save_flag == 3:
                            if key in cfg.defect_class_list:
                                save_flag = 2
                            else:
                                save_flag = 3

                for object_id in range(len(is_match_det)):
                    if is_match_det[object_id] == 0:
                        det_box = det_dicts[key][object_id]
                        save_dir = cfg.TEST.save_dir
                        name = os.path.basename(xml_name).split(
                            ".xml")[0] + "_" + str(object_id) + ".bmp"

                        dst_file = save_dir + '/detected_sample_is_error/' + key + "/"
                        if not os.path.exists(dst_file):
                            os.makedirs(dst_file)
                        dst_file += name

                        image_roi = image_cpy[
                            int(det_box[1]):int(det_box[3]), int(det_box[0]):int(det_box[2])]

                        if not os.path.exists(dst_file):
                            cv2.imwrite(dst_file, image_roi, [
                                int(cv2.IMWRITE_JPEG_QUALITY), 100])

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

        if save_flag >= 2:
            # show_gt_image
            xml_path = img_name.replace(cfg.image_postfix, ".xml")
            if os.path.exists(xml_path):
                # save_image_gt

                xml_image_name = os.path.basename(
                    img_name).replace(cfg.image_postfix, "_gt.bmp")
                save_image_gt(gt_image, xml_image_name, color_list, save_dir,
                              cfg.rename_class_list, xml_path, cfg.class_list)

        for boxes in results:
            for box in boxes:
                # for box_id, box in enumerate(boxes):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                score = float(box[4])
                label = int(box[5])
                w = x2 - x1
                h = y2 - y1
                label_name = cfg.class_list[label]

                if 0:  # 'KP' == label_name and max(w, h) <= 16:
                    image_roi = im[y1:y2, x1:x2]
                    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                    # (success, saliencyMap) = saliency.computeSaliency(
                    #     cv2.cvtColor(image_roi, cv2.COLOR_RGB2LAB))
                    # saliencyMap = (saliencyMap * 255).astype("uint8")
                    # threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                    #                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    image_gray = cv2.cvtColor(image_roi, cv2.COLOR_RGB2gray)
                    threshMap = cv2.threshold(image_gray, 0, 255,
                                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    # cv2.imwrite(save_dir + 'horizon/' +
                    #             str(box_id) + "_" + name.replace(".jpg", ".bmp"), image_roi, [
                    #                 int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    cv2.imwrite(save_dir + '/' + "A0_" + name, image_roi)
                    cv2.imwrite(save_dir + '/' + "A1_" + name, image_gray)
                    cv2.imwrite(save_dir + '/' + "A2_" + name, threshMap)

                    # if label_name != 'HH':
                    #     continue
                    # else:
                    #     save_flag = 1

                g, b, r = color_list[label_name][0]

                # label * 50, label * 100, label * 150
                if 1:  # label_name in ["HH"]:
                    # save_flag = 1
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (g, b, r), 1)
                    cv2.putText(draw_img, label_name, (x1 + w//2, y1 + h//2),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (g, b, r), 1)  # label*50, label*100, label*100
                    cv2.putText(draw_img, str(round(score, 3)), (x1 + w // 2 + 10, y1 + h // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (g, b, r), 1)  # label * 50, label * 100, label * 100

                # if save_xml:
                #  if 'M' == label_name or 'SL' == label_name or 'HH' == label_name:
                #    continue
                #  difficult = 0
                #  writer.addBndBox(x1, y1, x2, y2, label_name, difficult)
            if save_xml:
                writer.save(targetFile='./outputs/' +
                            name.replace(cfg.image_postfix, '.xml'))

        if len(results) > 0 and save_flag >= 2:
            name = os.path.basename(img_name)
            cv2.imwrite(save_dir + '/' + name, draw_img)

            name = os.path.basename(img_name).replace(
                cfg.image_postfix, "_undetected.bmp")

            if save_flag == 2:
                if not os.path.exists(save_dir + '/FN_defect/'):
                    os.makedirs(save_dir + '/FN_defect/')
                cv2.imwrite(save_dir + '/FN_defect/' + name, xml_image)
                cv2.imwrite(save_dir + '/' + name, xml_image)

                # hard_sample_dir = "/data/zhangcc/data/det_whiteshow_defect/1/hard_sample/"
                # shutil.copy(img_name,  hard_sample_dir +
                #             os.path.basename(img_name))
                # xml_name_cpy = img_name.replace(".bmp", ".xml")
                # shutil.copy(xml_name_cpy, hard_sample_dir +
                #             os.path.basename(xml_name_cpy))
            elif save_flag == 3:
                if not os.path.exists(save_dir + '/FN_device/'):
                    os.makedirs(save_dir + '/FN_device/')
                cv2.imwrite(save_dir + '/FN_device/' + name, xml_image)
                cv2.imwrite(save_dir + '/' + name, xml_image)

        else:
            # undetected_sample.append(name)
            # shutil.copyfile(img_name, save_dir + 'undetected_sample/' + name)
            # shutil.copyfile(img_name.replace(".jpg", ".xml"), save_dir + 'undetected_sample/' +
            #                 name.replace(".jpg", ".xml"))
            print('no object in this picture')

    return undetected_sample


if __name__ == "__main__":
    undetected_sample = main()
    print(undetected_sample)
