#!/usr/bin/python3
# -*- coding:utf8 -*-
# @Time    : 2019/4/4
# @Author  : Henson
# @File    : sys_data_distribution_xml.py

import codecs
import cv2
import os
import glob
import copy
import shutil
import matplotlib.pyplot as plt
import numpy as np

from chardet import detect
from tkinter import _flatten

import xml.etree.ElementTree as ET

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ratio = []
box_w = []
box_h = []
box_hw = []
widths = []
heights = []
avg_widths = []
avg_heights = []


def vis_ratio_and_size(boxes, image_size, idx, flag):
    input_size = 1024  # 160
    feat_stride = 8

    scale = (min(image_size[0], image_size[1]) * feat_stride) / input_size
    # image_size[0:2] = int(image_size[0:2] / scale)

    if flag:
        # ratio_hw = (int(boxes[3]) - int(boxes[1])) / \
        #     (int(boxes[2]) - int(boxes[0]))
        box_h_ = (int(boxes[3]) - int(boxes[1])) / scale
        box_w_ = (int(boxes[2]) - int(boxes[0])) / scale

        if image_size[0] > image_size[1]:
            ratio_hw = box_h_ / box_w_
        else:
            ratio_hw = box_w_ / box_h_

        ratio.append(ratio_hw)
        box_h.append(box_h_)
        box_w.append(box_w_)
        box_hw.append(box_h_)
        box_hw.append(box_w_)

        if idx == 0:
            # show and saveimage
            fig0 = plt.figure(0)
            plt.cla()
            # plt.scatter(range(len(ratio)), ratio, marker='o', color='r', s=15)
            plt.hist(ratio, bins=10, normed=0, facecolor="blue",
                     edgecolor="black", cumulative=True, alpha=0.7)
            # plt.show()
            # fig0.savefig(r'C:\Users\108890\Desktop\images\ratio.jpg')
            fig0.savefig('ratio.jpg')

            fig1 = plt.figure(1)
            plt.cla()
            # plt.scatter(range(len(box_h)), box_h, marker='o', color='r', s=15)
            plt.hist(box_h, bins=10, normed=0, facecolor="blue",
                     edgecolor="black", cumulative=True, alpha=0.7)
            # fig1.savefig(r'C:\Users\108890\Desktop\images\box_h.jpg')
            fig1.savefig('box_h.jpg')

            fig2 = plt.figure(2)
            plt.cla()
            # plt.scatter(range(len(box_w)), box_w, marker='o', color='r', s=15)
            plt.hist(box_w, bins=10, normed=0, facecolor="blue",
                     edgecolor="black", cumulative=True, alpha=0.7)
            # fig2.savefig(r'C:\Users\108890\Desktop\images\box_w.jpg')
            fig2.savefig('box_w.jpg')

            fig3 = plt.figure(3)
            plt.cla()
            plt.hist(box_hw, bins=10, normed=0, facecolor="blue",
                     edgecolor="black", cumulative=True, alpha=0.7)
            fig1.savefig('box_hw.jpg')

            # widths.append(image_size[0])
            # heights.append(image_size[1])
            # # avg_widths.append(sum(widths)/len(widths))
            # # avg_heights.append(sum(heights)/len(heights))

            # fig3 = plt.figure(3)
            # plt.cla()
            # plt.scatter(range(len(widths)), widths,
            #             marker='o', color='r', s=15)
            # fig3.savefig('w.jpg')

            # fig4 = plt.figure(4)
            # plt.cla()
            # plt.scatter(range(len(heights)), heights,
            #             marker='o', color='r', s=15)
            # fig4.savefig('h.jpg')


def changelabel(label):
    label = label.upper()
    # if 'SL' in label:
    #    label = 'SL'
    if 'HH' in label:
        label = 'HH'
    return label


def deleteDuplicatedElementFromList(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList


def all_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result


def add_dic(dic):
    _total = 0
    for key in dic:
        _total += dic[key]

    return _total


def find_post_name(name, post):
    return True if name in post else False


def filterxmls(xml_lists, str_='_new'):
    ll = []
    for xml in xml_lists:
        if find_post_name(str_ + '.xml', xml):
            ll.append(xml)
    return ll


def get_area(obj):
    bndbox = obj.find('bndbox')
    x1 = int(bndbox.find('xmin').text)
    y1 = int(bndbox.find('ymin').text)
    x2 = int(bndbox.find('xmax').text)
    y2 = int(bndbox.find('ymax').text)
    return (x2-x1)*(y2-y1)


def calc_data_distribution(input_file, copy_path, is_vis=0, postfix=".jpg"):
    xml_list = glob.glob(os.path.join(input_file, "**"), recursive=True)
    xml_list = filterxmls(xml_list, '')
    # print(xml_list)
    labels = []
    areas = {}
    for idx, (xml_name) in enumerate(xml_list):
        image_path = xml_name.replace(".xml", postfix)
        image = cv2.imread(image_path, -1)
        if image is None:
            continue

        print("{}/{}, {}".format(idx, len(xml_list), xml_name))
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_name, parser=parser)
        root = tree.getroot()
        labels_per_image = []
        for obj_id, obj in enumerate(root.findall('object')):
            boxes = []
            boxes.append(int(obj.find('bndbox/xmin').text))
            boxes.append(int(obj.find('bndbox/ymin').text))
            boxes.append(int(obj.find('bndbox/xmax').text))
            boxes.append(int(obj.find('bndbox/ymax').text))

            class_name = obj.find('name').text.strip()
            class_name = class_name.upper()

            bh = boxes[3] - boxes[1]
            bw = boxes[2] - boxes[0]
            if 1:  # 'KP' in class_name and (bh <= 16 or bw <= 16):
                # [ymin:ymax, xmin:xmax]
                image_roi = image[boxes[1]:boxes[3], boxes[0]:boxes[2]]
                new_image_path = os.path.join(input_file, "vis", class_name)
                if not os.path.exists(new_image_path):
                    os.makedirs(new_image_path)

                new_image_path = os.path.join(new_image_path, os.path.basename(
                    xml_name).split(".xml")[0] + "_" + str(obj_id) + ".bmp")
                cv2.imwrite(new_image_path, image_roi, [
                            int(cv2.IMWRITE_JPEG_QUALITY), 100])

                # remove file included "SL"
                if copy_path is not None:
                    if 0 == obj_id:
                        copy_flag = 1
                    if "SL" == class_name:
                        copy_flag = 0
                    if len(root.findall('object')) - 1 == obj_id and copy_flag:
                        if not os.path.exists(copy_path):
                            os.makedirs(copy_path)

                        shutil.copyfile(image_path, os.path.join(
                            copy_path, os.path.basename(image_path)))
                        shutil.copyfile(xml_name, os.path.join(
                            copy_path, os.path.basename(xml_name)))

            image_size = []
            image_size.append(int(root.find("size/width").text))
            image_size.append(int(root.find("size/height").text))

            vis_ratio_and_size(boxes, image_size, obj_id, is_vis)

            label = obj.find('name').text
            label = changelabel(label)
            if not label in areas.keys():
                areas[label] = []
            area = get_area(obj)
            areas[label].append(area)
            labels_per_image.append(label)
        labels.append(labels_per_image)
    label_freq = all_list(list(_flatten(labels)))
    # print(len(xml_list), label_freq)
    info_label = [add_dic(label_freq), copy.deepcopy(label_freq)]

    for k in label_freq:
        label_freq[k] = 0
    # print(label_freq)

    # print(labels)
    for lb in labels:
        # print(lb)
        label_dic = all_list(lb)
        # print(label_dic)
        for key in label_dic:
            # print(key, label_dic[key])
            label_freq[key] += 1
            # print(label_freq)
            # break

    # print(len(xml_list), label_freq)
    info_image = [len(xml_list), label_freq]  # add_dic(label_freq),

    idx = 1
    CATEGORIES = []
    CATEGORIES_ID = []
    PRE_DEFINE_CATEGORIES = copy.deepcopy(label_freq)
    for key in label_freq:
        CATEGORIES.append(key)
        CATEGORIES_ID.append(idx)
        PRE_DEFINE_CATEGORIES[key] = idx
        idx = idx + 1

    return info_label, info_image, areas, PRE_DEFINE_CATEGORIES, CATEGORIES, CATEGORIES_ID


def maxrange(maxarea):
    bins = np.array([0, 36, 42, 450, 1800, 5000, 20000,
                     np.iinfo(np.int32).max], dtype=int)
    sort_ = (bins/maxarea) > 1.0
    max_index = np.where(sort_)[0][0]
    new_bins = np.zeros((max_index), dtype=int)
    new_bins[0:max_index - 1] = bins[0:max_index-1]
    new_bins[max_index - 1] = maxarea
    return new_bins


def drawArea(areas):
    # plt.figure('')
    for key in areas:
        label_areas = np.array(areas[key], dtype=np.int)
        bins = maxrange(np.max(label_areas))
        # label_areas = np.sort(label_areas)
        plt.hist(label_areas,
                 bins=bins,
                 # normed = True,
                 cumulative=False,
                 color='steelblue',
                 edgecolor='k',
                 label='hist for area / frequency')
        plt.title(key)
        plt.xlabel('area')
        plt.ylabel('frequency')
        plt.tick_params(top='off', right='off')
        plt.legend(loc='best')
        plt.show()
        # plt.savefig(r'G:\data' +'/' + key +'.png')


def remove_xml_chinese_character(input_path):
    file_paths = glob.glob(os.path.join(input_path, "*.xml"))
    for file_path in file_paths:
        if os.path.basename(file_path).find("黑色") > 0:
            f_new = open(file_path.replace(".xml", "_new.xml").replace(
                "黑色", "black"), "w", encoding='utf-8')
            # shutil.copy(file_path, file_path.replace("黑色", "black"))
            shutil.copy(file_path.replace(".xml", ".jpg"), file_path.replace(
                ".xml", "_new.jpg").replace("黑色", "black"))
        else:
            f_new = open(file_path.replace(
                ".xml", "_new.xml"), "w", encoding='utf-8')
            # shutil.copy(file_path, file_path.replace(".xml", "_new.jpg"))
            shutil.copy(file_path.replace(".xml", ".jpg"),
                        file_path.replace(".xml", "_new.jpg"))

        # f_new = open(file_path.replace(".xml", "_new.xml"), "w") #.replace("black", "黑色")
        print(file_path)

        f = open(file_path, 'rb')
        for line in f.readlines():
            try:
                line = line.decode('utf-8')
            except:
                line = line.decode('GB2312')

            line = line.rstrip()
            if line.find("<folder>") > 0:
                f_new.write("\t<folder></folder>\n")
            elif line.find("<filename>") > 0:
                if os.path.basename(file_path).find("黑色") > 0:
                    f_new.write("\t<filename>{}</filename>\n".format(os.path.basename(
                        file_path).replace(".xml", "_new.jpg").replace("黑色", "black")))
                else:
                    f_new.write("\t<filename>{}</filename>\n".format(
                        os.path.basename(file_path).replace(".xml", "_new.jpg")))
            elif line.find("<path>") > 0:
                f_new.write("\t<path></path>\n")
            else:
                f_new.write(str(line) + "\n")

        f.close()
        f_new.close()

        # os.remove(file_path)
        # os.remove(file_path.replace(".xml", ".jpg"))


if __name__ == '__main__':
    # r"\\10.180.2.101\zcc-data\det_vertical_defect\origin_and_argument" #det_vertical_defect mobile_terminal_screen_defect
    # input_file = "/data/zhangcc/data/det_vertical_defect/2/original"
    input_file = "/data/zhangcc/data/det_whiteshow_defect/1/hard_sample"
    copy_path = None  # "/data/zhangcc/data/det_center_defect/2/original"
    # remove_xml_chinese_character(input_file)

    info_label, info_image, areas, PRE_DEFINE_CATEGORIES, CATEGORIES, CATEGORIES_ID = calc_data_distribution(
        input_file, copy_path, 1, ".bmp")
    print("info_label: ", info_label)
    print("info_image: ", info_image)
    print("PRE_DEFINE_CATEGORIES: ", PRE_DEFINE_CATEGORIES)
    print("CATEGORIES: ", CATEGORIES)
    print("CATEGORIES_ID: ", CATEGORIES_ID)
    # drawArea(areas)
