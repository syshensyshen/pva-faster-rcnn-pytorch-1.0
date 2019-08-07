from __future__ import print_function

import numpy as np
import json
import os

import torch

from models.config import cfg
# rom lib.roi_layers.nms import nms
from torchvision.ops import nms


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

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = ((ws >= min_size) & (hs >= min_size))
    return keep

def _get_detections(dataset, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    model.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            img = data['img']
            im_scales = np.array([np.hstack((img.size(0), img.size(1), scale, scale))],
                                 dtype=np.float32)
            gt_boxes = np.zeros((1, 1, 5), dtype=np.float32)
            img = img.view(-1, img.size(0), img.size(1), img.size(2))

            # run network

            rois, cls_prob, bbox_pred = model(img.permute(0, 3, 1, 2).cuda(), torch.from_numpy(im_scales).cuda(),
                                              torch.from_numpy(gt_boxes).cuda())
            std = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
            mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
            std = torch.from_numpy(std).cuda()
            mean = torch.from_numpy(mean).cuda()
            pred_boxes = bbox_transform_inv(rois[:, :, 1:5], bbox_pred, 1, std, mean)
            scores = cls_prob
            detections = []

            for label in range(1, dataset.num_classes()):
                cls_scores = scores[0, :, label]
                scores_over_thresh = (cls_scores > score_threshold)
                cls_keep = cls_scores[scores_over_thresh]
                bboxes_keep = pred_boxes[0, scores_over_thresh, label * 4:(label + 1) * 4]
                filter_keep = _filter_boxes(bboxes_keep, 8)
                cls_keep = cls_keep[filter_keep]
                bboxes_keep = bboxes_keep[filter_keep, :]
                keep_idx_i = nms(bboxes_keep, cls_keep, 0.9)
                keep_idx_i = keep_idx_i.long().view(-1)
                bboxes_keep = bboxes_keep[keep_idx_i, :]
                cls_keep = cls_keep[keep_idx_i]
                bboxes_keep[:, 0] /= scale
                bboxes_keep[:, 1] /= scale
                bboxes_keep[:, 2] /= scale
                bboxes_keep[:, 3] /= scale
                if bboxes_keep.size(0) > 0:
                    result = np.zeros((bboxes_keep.size(0), 6), dtype=np.float32)
                    result[:, 0:4] = bboxes_keep.cpu()
                    result[:, 4] = cls_keep.cpu()
                    result[:, 5] = label
                    all_detections[index][label] = result
                else:
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.get_annotation(i)
        annotations = annotations[1]

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            index = int(label)            
            all_annotations[i][index] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.35,
    score_threshold=0.3,
    max_detections=200,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    #print(all_detections.shape, all_annotations.shape)
    
    average_precisions = {}

    for label in range(generator.num_classes()):
        #print('generator.num_classes(): ', generator.num_classes())
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            #print(i)
            #print(detections.shape)
            #print(annotations.shape)

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    
    print('\nmAP:')
    for label in range(generator.num_classes()):
        #print('generator.num_classes() is: ', generator.num_classes())
        #print('label index is: ', label)              
        label_name = generator.label_to_name(label)              
        #print('label_name: ', label_name)              
        print('{}: {}'.format(label_name, average_precisions[label][0]))
    
    return average_precisions