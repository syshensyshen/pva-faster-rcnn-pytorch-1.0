import os
import sys
import numpy as np
from glob import glob
import copy
import cv2
import random
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element,ElementTree

from cvs_xml_tools import csvToxml
import argument_tools as argument_tools

def check_file(path, postfix):
    return os.path.isfile(path + postfix)

def save_data(argument, img, tree, img_name, postfix):
    flie_path = os.path.join(argument_path, argument)
    if not os.path.exists(flie_path):
        os.mkdir(flie_path)
    if postfix == '.png':
        cv2.imwrite(os.path.join(flie_path, os.path.basename(img_name).replace(postfix, '_' + argument +postfix)), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif postfix == '.jpg':
        cv2.imwrite(os.path.join(flie_path, os.path.basename(img_name).replace(postfix, '_' + argument +postfix)), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(os.path.join(flie_path, os.path.basename(img_name).replace(postfix, '_' + argument +postfix)), img)
    tree.write(os.path.join(flie_path, os.path.basename(img_name).replace(postfix, '_' + argument +'.xml')), encoding='utf-8', xml_declaration=True)

def ganerate_dataset(csv_file, img_path, xml_output_path, argument_path):

    #csvToxml(csv_file, img_path, xml_output_path)

    #argument_list = ['rotation', 'filp', 'crop', 'noise', 'blur', 'light']
    argument_list = ['flip']

    original_datasets = glob(xml_output_path + '/*.xml')

    for xml in original_datasets:
        #xml = r'G:\data\show\white\original_0529\350-B-NDJ-20181211FAC-201904211313131570-100-middle.xml'
        tree = ET.parse(xml)
        # root = tree.getroot()
        # objs = root.findall('object')
        # if len(objs) == 0:
        #     print(xml)
        img_name = os.path.join(img_path, os.path.basename(xml).replace('.xml', ''))
        postfix = ''
        if check_file(img_name, '.jpg'):
            postfix = '.jpg'
        if check_file(img_name, '.bmp'):
            postfix = '.bmp'
        if check_file(img_name, '.png'):
            postfix = '.png'
        # if check_file(img_name, '.JPG'):
        #     postfix = '.JPG'
        img_name += postfix
        
        img = cv2.imread(img_name)

        if  'flip' in argument_list:
            flip_codes = [np.random.randint(-1, 2)]
            for flip_code in flip_codes:
                img_flip, tree_flip = argument_tools.flip(img.copy(), copy.deepcopy(tree), flip_code)
                save_data('flip_'+str(flip_code).replace('-1', '_1'), img_flip, tree_flip, img_name, postfix)
        if 'rotation' in argument_list:            
            img_r, tree_r = argument_tools.rotation(img.copy(), copy.deepcopy(tree), random.randint(2, 10))
            save_data('rotation', img_r, tree_r, img_name, postfix)
        if 'crop' in argument_list:            
            img_c, tree_c = argument_tools.crop(img.copy(), copy.deepcopy(tree), random.randint(1, 25) / 100, 8)
            save_data('crop', img_c, tree_c, img_name, postfix)
        if 'noise' in argument_list:            
            img_n, tree = argument_tools.noise(img.copy(), copy.deepcopy(tree))     
            save_data('noise', img_n, tree, img_name, postfix)       
        if 'blur' in argument_list:            
            img_b, tree_b = argument_tools.blur(img.copy(), copy.deepcopy(tree)) 
            save_data('blur', img_b, tree_b, img_name, postfix)           
        if 'light' in argument_list:
            img_l, tree_l = argument_tools.light(img.copy(), tree)
            save_data('light', img_l, tree_l, img_name, postfix)

if __name__ == "__main__":
    csv_file = "" #r"G:\competition\HKcar\data.csv"	
    img_path = "/data/zhangcc/data/det_vertical_defect/2/original"
    xml_output_path = img_path
    argument_path = r'/data/zhangcc/data/det_vertical_defect/2/argument'
    ganerate_dataset(csv_file, img_path, xml_output_path, argument_path)