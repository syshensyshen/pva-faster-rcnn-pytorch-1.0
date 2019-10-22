import os
import sys
import cv2
import csv
import random
import pandas
import shutil
import math
import numpy as np
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element,ElementTree

import matplotlib.pyplot as plt

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

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generateXML(img_name, shape, boxes, output, class_name = 'hkcar'):
	xml_name = os.path.basename(img_name)
	
	root = ET.Element("annotation")

	ET.SubElement(root, "CreateVersion").text = str(2.5)
	ET.SubElement(root, "folder").text = " "

	#image_name
	ET.SubElement(root, "filename").text = img_name
	ET.SubElement(root, "path").text = " "

	source_node = ET.SubElement(root, "source")
	ET.SubElement(source_node, "database").text = "Unknown"

	ET.SubElement(root, "score").text = str(0)

	#image_size
	size_node = ET.SubElement(root, "size")
	
	ET.SubElement(size_node, "width").text = str(shape[0])
	ET.SubElement(size_node, "height").text = str(shape[1])
	ET.SubElement(size_node, "depth").text = str(shape[2])
	
	ET.SubElement(root, "segmented").text = str(0)

	for i in range(len(boxes)):
		#add object
		object = ET.SubElement(root, "object")
		ET.SubElement(object, "name").text = class_name
		ET.SubElement(object, "pose").text = str(0)
		ET.SubElement(object, "truncated").text = str(0)
		ET.SubElement(object, "difficult").text = str(0)
		ET.SubElement(object, "staintype").text = " "
		ET.SubElement(object, "level").text = str(1)
		
		object_bndbox_node = ET.SubElement(object, "bndbox")
		ET.SubElement(object_bndbox_node, "xmin").text = str(boxes[i][0])
		ET.SubElement(object_bndbox_node, "ymin").text = str(boxes[i][1])
		ET.SubElement(object_bndbox_node, "xmax").text = str(boxes[i][2])
		ET.SubElement(object_bndbox_node, "ymax").text = str(boxes[i][3])

		object_shape_node = ET.SubElement(object, "shape")
		shape_points_node = ET.SubElement(object_shape_node, "points")
		shape_points_node.attrib = {"type":"rect","color":"Red","thickness":"3"}
		
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][0])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][1])
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][2])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][3])
	
	tree = ET.ElementTree(root)

	indent(root)

	tree.write(os.path.join(output, xml_name.replace('.jpg', '.xml')))

def parse_csv(csv_file_name, image_path, xml_output_path):
	df = pandas.read_csv(csv_file_name)
	
	boxes = []
	image_id = 0
	image_name = os.path.basename(df['filename'][0])
	image_size = [0, 0, 0]
	for i in range(len(df['filename'])):
		if image_name == os.path.basename(df['filename'][i]):
			if image_name == "end":
				print("image_name is not exist...")
				continue
			image = cv2.imread(os.path.join(image_path, image_name), -1)
			if image is None:
				print("image is NULL...")
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append([df['xmin'][i], df['ymin'][i], df['xmax'][i], df['ymax'][i]])
		else:
			image_id = image_id + 1			
			
			generateXML(image_name, image_size, boxes, xml_output_path)
			
			boxes = []
			image_name = os.path.basename(df['filename'][i])
			if image_name == "end":
				print("image_name is not exist...")
				continue
			image = cv2.imread(os.path.join(image_path, image_name), -1)
			if image is None:
				print("image is NULL...")
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append([df['xmin'][i], df['ymin'][i], df['xmax'][i], df['ymax'][i]])

def csvToxml(csv_file, img_path, xml_output_path):
	#csv_file = r"G:\competition\HKcar\data.csv"
	
	#image_path = r"G:\competition\HKcar"
	#xml_output_path = r"G:\competition\HKcar/Annotations"
	
	checkpath(img_path)
	checkpath(xml_output_path)
		
	parse_csv(csv_file, img_path, xml_output_path)

#if __name__ == "__main__":
	#csvToxml()