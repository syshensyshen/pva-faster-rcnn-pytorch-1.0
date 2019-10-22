#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import cv2
#import csv
import random
#import pandas
import shutil
import math
import numpy as np
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element,ElementTree
import matplotlib.pyplot as plt
from sample_argument_tools import generatorNoise, generatorBlur, lightexchange
from tqdm import tqdm

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))
    shift_w = ((bound_w / 2) - image_center[0])
    shift_h = ((bound_h / 2) - image_center[1])
    rotation_mat[0, 2] += shift_w
    rotation_mat[1, 2] += shift_h

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, shift_w, shift_h

def rotate_point(x0, y0, center_x, center_y, angle, bound_w, bound_h):
    delta_x = x0 - center_x
    delta_y = y0 - center_y
    small_shift = random.random() * 3
    angle_sin = math.sin(angle + small_shift * (np.pi / 180.0))
    angle_cos = math.cos(angle + small_shift * (np.pi / 180.0))			
    x = delta_x * angle_cos + delta_y * angle_sin + center_x
    y = -delta_x * angle_sin + delta_y * angle_cos + center_y
    x = math.floor(x + bound_w)
    y = math.floor(y + bound_h)

    return x, y

def rotation(img, tree, angle):
    
    center_x = img.shape[1] >> 1
    center_y = img.shape[0] >> 1
    image_RT, bound_w, bound_h = rotate_image(img.copy(), angle)   
    root = tree.getroot()
    for obj in root.findall('object'):
        xmin = int(int(obj.find('bndbox/xmin').text) / 1)
        ymin = int(int(obj.find('bndbox/ymin').text) / 1)
        xmax = int(int(obj.find('bndbox/xmax').text) / 1)
        ymax = int(int(obj.find('bndbox/ymax').text) / 1)
        x00 = xmin
        y00 = ymin
        x11 = xmax
        y11 = ymax
        t_x00, t_y00 = rotate_point(x00, y00, center_x, center_y, angle, bound_w, bound_h)
        t_x01, t_y01 = rotate_point(x00, y11, center_x, center_y, angle, bound_w, bound_h)
        t_x10, t_y10 = rotate_point(x11, y00, center_x, center_y, angle, bound_w, bound_h)			
        t_x11, t_y11 = rotate_point(x11, y11, center_x, center_y, angle, bound_w, bound_h)
        t_xarray = [t_x00, t_x01, t_x10, t_x11]
        t_yarray = [t_y00, t_y01, t_y10, t_y11]        
        t_xmin = int(min(t_xarray))
        t_ymin = int(min(t_yarray))
        t_xmax = int(max(t_xarray))
        t_ymax = int(max(t_yarray))
        obj.find('bndbox/xmin').text = str(t_xmin)
        obj.find('bndbox/ymin').text = str(t_ymin)
        obj.find('bndbox/xmax').text = str(t_xmax)
        obj.find('bndbox/ymax').text = str(t_ymax)

    return image_RT, tree

def flip(img, tree, code=-1):
    img_flip = cv2.flip(img, code)
    root = tree.getroot()
    for obj in root.findall('object'):
        xmin = int(int(obj.find('bndbox/xmin').text) / 1)
        ymin = int(int(obj.find('bndbox/ymin').text) / 1)
        xmax = int(int(obj.find('bndbox/xmax').text) / 1)
        ymax = int(int(obj.find('bndbox/ymax').text) / 1)        
        if -1 == code:#hv
            t_xmin = img.shape[1] - 1 - xmax
            t_ymin = img.shape[0] - 1 - ymax
            t_xmax = img.shape[1] - 1 - xmin
            t_ymax = img.shape[0] - 1 - ymin
        elif 0 == code:#v
            t_xmin = xmin 
            t_ymin = img.shape[0] - 1 - ymax
            t_xmax = xmax
            t_ymax = img.shape[0] - 1 - ymin
        elif 1 == code:#h
            t_xmin = img.shape[1] - 1 - xmax 
            t_ymin = ymin
            t_xmax = img.shape[1] - 1 - xmin
            t_ymax = ymax
        t_xmin = t_xmin if t_xmin >= 0 else 0
        t_ymin = t_ymin if t_ymin >= 0 else 0
        t_xmax = t_xmax if t_xmax < img.shape[1] else img.shape[1]
        t_ymax = t_ymax if t_ymax < img.shape[0] else img.shape[0]
        obj.find('bndbox/xmin').text = str(t_xmin)
        obj.find('bndbox/ymin').text = str(t_ymin)
        obj.find('bndbox/xmax').text = str(t_xmax)
        obj.find('bndbox/ymax').text = str(t_ymax)

    return img_flip, tree

def crop(img, tree, crop_ratio, min_size=1):
    crop_size_x = int(img.shape[1] * crop_ratio)
    crop_size_y = int(img.shape[0] * crop_ratio)    

    img_crop = img[crop_size_y//2:img.shape[0]-crop_size_y//2, crop_size_x//2:img.shape[1]-crop_size_x//2, :]
    root = tree.getroot()
    objs = root.findall('object')
    for obj in objs:
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        origin_h = ymax - ymin
        origin_w = xmax - xmin

        xmin -= crop_size_x/2
        ymin -= crop_size_y/2
        xmax -= crop_size_x/2
        ymax -= crop_size_y/2
        
        xmin = xmin if xmin >= 0 else 0
        xmin = xmin if xmin <= img_crop.shape[1] - 1 else img_crop.shape[1] - 1
        ymin = ymin if ymin >= 0 else 0
        ymin = ymin if ymin <= img_crop.shape[0] - 1 else img_crop.shape[0] - 1
        xmax = xmax if xmax >= 0 else 0
        xmax = xmax if xmax <= img_crop.shape[1] - 1 else img_crop.shape[1] - 1
        ymax = ymax if ymax >= 0 else 0
        ymax = ymax if ymax <= img_crop.shape[0] - 1 else img_crop.shape[0] - 1
        h = ymax - ymin
        w = xmax - xmin
        remove = False
        if (origin_h+1e-3) / (h+1e-3) > 1.2 and h <= min_size:
            remove = True
        if (origin_w+1e-3) / (w+1e-3) > 1.2 and w <= min_size:
            remove = True
        if remove:
            root.remove(obj)
                

        obj.find('bndbox/xmin').text = str(int(xmin))
        obj.find('bndbox/ymin').text = str(int(ymin))
        obj.find('bndbox/xmax').text = str(int(xmax))
        obj.find('bndbox/ymax').text = str(int(ymax))

    return img_crop, tree

def noise(img, tree):
    img = generatorNoise(img, np.random.randint(30, 300))

    return img, tree

def blur(img, tree):
    kernel_size = random.randint(3, 6)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = random.randint(3, 7)
    img = generatorBlur(img, kernel_size, sigma)
    return img, tree

def light(img, tree):
    illumination_img = lightexchange(img)
    return illumination_img, tree

