# -*- coding: utf-8 -*-
'''
# @Author  : syshen 
# @date    : 2019/1/7-2019/1/23
# @File    : data loader
'''
import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import uuid
import cv2
import random

#SCALES = (416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024)
#SCALES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864)
SCALES = (576, 608, 640)
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
#PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
#PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])
SCALE_MULTIPLE_OF = 32
MAX_SIZE = 1440


#class_list = ('__background__', 'left', 'right', 'top','bottom', 'middle', 'back', 'label', 'lable')
#class_list = ('__background__', # always index 0
#                         'aeroplane', 'bicycle', 'bird', 'boat',
#                         'bottle', 'bus', 'car', 'cat', 'chair',
#                         'cow', 'diningtable', 'dog', 'horse',
#                         'motorbike', 'person', 'pottedplant',
#                         'sheep', 'sofa', 'train', 'tvmonitor',
#                         'rebar')
#class_list = ('__background__', 'HM', 'TT')
class_list = ('__background__', 'left', 'top', 'middle', 'back', 'label')
class_to_ind = dict(zip(class_list, range(0, len(class_list))))

def load_pascal_annotation(xml_path):
    if not os.path.isfile(xml_path):
        print('xml file not exists!\r\n')
    tree = ET.parse(xml_path)
    img_name = tree.find('filename').text.strip()
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, len(class_list)), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        #print(obj.find('name').text.strip)
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) #- 1
        y1 = float(bbox.find('ymin').text) #- 1
        x2 = float(bbox.find('xmax').text) #- 1
        y2 = float(bbox.find('ymax').text) #- 1
        #cls = self._class_to_ind[obj.find('name').text.lower().strip()]        
        class_name = obj.find('name').text.strip()
        if class_name == 'bottom':
            class_name = 'top'
        if class_name == 'right':
            class_name = 'left'
        if class_name == 'lable':
            class_name = 'label'
        if not class_name in class_list:
            continue
        cls = class_to_ind[class_name]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)
    zeros_label = []
    for ix, label in enumerate(gt_classes):
        if label == 0:
            zeros_label.append(ix)
    #print boxes, gt_classes
    boxes = np.delete(boxes, zeros_label, axis=0)
    gt_classes = np.delete(gt_classes, zeros_label, axis=0)
    #print(boxes, gt_classes)
    #raw_input()

    return {'img_name': img_name,
            'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False,
            'seg_areas' : seg_areas}

def prep_im_for_blob(im, pixel_means, target_size, max_size, multiple):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
    im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)
    return im, np.array([im_scale_x, im_scale_y])

def im_list_to_blob(im):
   
    blob = np.zeros((1, im.shape[0], im.shape[1], 3),
                    dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    #print(blob[0, 0:im.shape[0], 0:im.shape[1], :])
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _get_image_blob(im, PIXEL_MEANS, target_size, MAX_SIZE, SCALE_MULTIPLE_OF):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    #from sample_argument_tools import lightexchange
    #im = cv2.imread(img_path)
    im, im_scale = prep_im_for_blob(im, PIXEL_MEANS, target_size,
                                        MAX_SIZE, SCALE_MULTIPLE_OF)
    # Create a blob to hold the input images
    blob = np.zeros((im.shape[0], im.shape[1], 3),
                    dtype=np.float32)
    blob[0:im.shape[0], 0:im.shape[1], :] = im
    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (2, 0, 1)
    blob = blob.transpose(channel_swap)
    #print(blob.shape)

    return blob, im_scale 

def get_target_size(target_size, im, multiple, max_size):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if len(im_shape) > 2:
        channles = im_shape[2]
    else:
        channles = 1
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    width = np.floor(im.shape[1] * im_scale / multiple) * multiple 
    height = np.floor(im.shape[0] * im_scale / multiple) * multiple 
    return int(width), int(height), channles

def check_file(path, name, postfix):
    return os.path.isfile(path + '/' + name.replace('.xml', postfix))

def prepareBatchData(xml_path, img_path, batch_size, xmllist):
    #for ix, xml_path in enumerate(xmllist):
    ims = []
    boxes = []
    labels = []
    max_len = 0
    for xml in xmllist:
        name = os.path.basename(xml)
        xml_context = load_pascal_annotation(xml_path + '/' + name)
        if max_len < xml_context['boxes'].shape[0]:
            max_len = xml_context['boxes'].shape[0]
        postfix = ''
        if check_file(img_path, name, '.jpg'):
            postfix = '.jpg'
        if check_file(img_path, name, '.bmp'):
            postfix = '.bmp'
        if check_file(img_path, name, '.png'):
            postfix = '.png'
        im = cv2.imread(img_path + '/' + name.replace('.xml', postfix))
        #print(img_path, name, postfix)
        #print(im.shape)
        ims.append(im)
        boxes.append(xml_context['boxes'])
        labels.append(xml_context['gt_classes'])
    gt_boxes = np.zeros((batch_size, max_len, 5), dtype=np.float32)    
    im_scales = np.zeros((batch_size, 4), dtype = np.float32)
    target_size = SCALES[random.randint(0, len(SCALES)-1)]
    #print(len(ims))
    width, height, channles = get_target_size(target_size, ims[random.randint(0, len(ims)-1)], SCALE_MULTIPLE_OF, MAX_SIZE)
    im_blobs = np.zeros((batch_size, channles, height, width), dtype = np.float32)
    for ix, gt in enumerate(boxes):
        len_t = gt.shape[0]
        im_scale_x = width / float(ims[ix].shape[1])
        im_scale_y = height / float(ims[ix].shape[0])
        im_scale = np.array([im_scale_x, im_scale_y])
        im = ims[ix]
        im = im.astype(np.float32)
        #im = (im - PIXEL_MEANS) / PIXEL_STDS
        #im = im - PIXEL_MEANS
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
        im_blobs[ix, :, :, :] = im_list_to_blob(im)
        gt_boxes[ix, 0:len_t, 0] = gt[:, 0] * im_scale[0]
        gt_boxes[ix, 0:len_t, 1] = gt[:, 1] * im_scale[1]
        gt_boxes[ix, 0:len_t, 2] = gt[:, 2] * im_scale[0]
        gt_boxes[ix, 0:len_t, 3] = gt[:, 3] * im_scale[1]   
        gt_boxes[ix, 0:len_t, 4:5] = labels[ix].reshape(len_t, 1)
        im_scales[ix,:] = np.array(
            [np.hstack((height, width, im_scale[0], im_scale[1]))],
            dtype=np.float32)
        #print(im_blobs.shape)
    
    return [gt_boxes, im_blobs, im_scales]