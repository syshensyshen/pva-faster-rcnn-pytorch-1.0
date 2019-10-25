from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

# import skimage.io
# import skimage.transform
# import skimage.color
# import skimage
import cv2
from IPython import embed
from PIL import Image
from six import raise_from

from glob import glob
import pathlib
import xml.etree.ElementTree as ET
import cv2


def check_file(path, postfix):
    return os.path.isfile(path + postfix)


def get_target_size(target_size, im_shape, multiple, max_size):
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    width = np.floor(im_shape[1] * im_scale / multiple) * multiple
    height = np.floor(im_shape[0] * im_scale / multiple) * multiple
    return int(width), int(height),


def make_scale_anno(annot, im_scale_x, im_scale_y):
    annots = []
    for anno in annot:
        annotation = np.zeros((1, 5))
        annotation[0, 0] = anno[0, 0] * im_scale_x  # x1
        annotation[0, 0] = anno[0, 1] * im_scale_y  # y1
        annotation[0, 0] = anno[0, 2] * im_scale_x  # x2
        annotation[0, 0] = anno[0, 3] * im_scale_y  # y2
        annotation[0, 0] = anno[0, 4]
        annots.append(annotation)

    return annots


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations',
                                      'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images',
                            self.set_name, image_info['file_name'])
        # img = skimage.io.imread(path)
        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        img = cv2.imread(path)
        if len(img.shape) == 2:
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(
                    csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError(
                'invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(
                    csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError(
                'invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(
                class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError(
                    'line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot,
                  'filename': self.image_names[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        # img = skimage.io.imread(self.image_names[image_index])
        img = cv2.imread(self.image_names[image_index])

        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                    None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(
                x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(
                y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(
                x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(
                y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError(
                    'line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError(
                    'line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(
                    line, class_name, classes))

            result[img_file].append(
                {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class XML_VOCDataset:

    def __init__(self, cfg, img_path, xml_path, class_list, transform=None):
        """
        Dataset for VOC data labeled by VOC format.
        """
        self.cfg = cfg
        self.img_root = img_path
        self.xml_root = xml_path
        self.transform = transform
        self.imgs = glob(os.path.join(img_path, "*.jpg"))
        self.xmls = glob(os.path.join(xml_path, "*.xml"))
        self.class_list = tuple(class_list)
        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_list)}

    def label_to_name(self, label):
        return self.class_list[label]

    def __getitem__(self, index):
        xml_path = self.xmls[index]
        name = os.path.basename(xml_path)
        img_path = os.path.join(self.img_root, name.replace('.xml', '.jpg'))
        annot = self._get_annotation(xml_path)
        img = self.get_image(index)
        sample = {'img': img, 'annot': annot, 'filename': img_path}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_image(self, index):
        xml_path = self.xmls[index]
        name = os.path.basename(xml_path)
        img_path = os.path.join(self.img_root, name.replace('.xml', ''))
        postfix = ''
        if check_file(img_path, '.jpg'):
            postfix = '.jpg'
        if check_file(img_path, '.bmp'):
            postfix = '.bmp'
        if check_file(img_path, '.png'):
            postfix = '.png'
        if check_file(img_path, '.JPG'):
            postfix = '.JPG'

        img_path += postfix
        image = cv2.imread(img_path)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image  # image.astype(np.float32) #/ 255.0
        # return image.astype(np.float32) #/ 255.0

    def get_annotation(self, index):
        xml_path = self.xmls[index]
        return xml_path, self._get_annotation(xml_path)

    def __len__(self):
        return len(self.xmls)

    def num_classes(self):
        return len(self.class_list)

    def _get_annotation(self, annotation_file):
        objects = ET.parse(annotation_file).findall("object")
        annotations = np.zeros((0, 5))

        for object in objects:
            class_name = object.find('name').text.strip()
            # class_name = class_name.upper()
            # print(class_name, self.class_list)
            bbox = object.find('bndbox')

            if 'TZ' in class_name:
                class_name = 'TZ'
            if 'FN' in class_name:
                class_name = 'FN'
            if class_name == 'LB':
                class_name = 'LD'

            if not class_name in self.class_list:
                continue

            # if not class_name in ["SL"]:
            #     continue

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text)  # - 1
            y1 = float(bbox.find('ymin').text)  # - 1
            x2 = float(bbox.find('xmax').text)  # - 1
            y2 = float(bbox.find('ymax').text)  # - 1

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            h = y2 - y1
            w = x2 - x1

            if w < 5 or h < 5:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            annotation[0, 4] = self.class_dict[class_name]
            annotations = np.append(annotations, annotation, axis=0)
            # print(annotations)
            # print(class_name)
        return annotations

    def __get_image_shape(self, xml_path):
        shape = ET.parse(xml_path).find("size")
        width = shape.find('width').text
        height = shape.find('height').text
        return int(height), int(width)

    def image_aspect_ratio(self, image_index):
        xml_path = self.xmls[image_index]
        height, width = self.__get_image_shape(xml_path)
        if height > 0 and width > 0:
            return float(width) / float(height)
        name = os.path.basename(xml_path)
        img_path = os.path.join(self.img_root, name.replace('.xml', ''))
        postfix = ''
        if check_file(img_path, '.jpg'):
            postfix = '.jpg'
        if check_file(img_path, '.bmp'):
            postfix = '.bmp'
        if check_file(img_path, '.png'):
            postfix = '.png'
        if check_file(img_path, '.JPG'):
            postfix = '.JPG'

        img_path += postfix

        # image = Image.open(img_path)
        image = cv2.imread(img_path)
        return float(image.shape[1]) / float(image.shape[0])


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[0]) for s in imgs]
    batch_size = len(imgs)

    shapes = [shape for shape in range(1088, 1440, 32)]
    min_side = shapes[random.randint(0, len(shapes) - 1)]

    width, height = get_target_size(min_side, (heights[0], widths[0]), 32, 2400)

    padded_imgs = torch.zeros(batch_size, height, width, 3)
    im_info = torch.zeros(batch_size, 4)

    for i in range(batch_size):
        img = imgs[i]
        resize_img = cv2.resize(img, (width, height))
        # normal_img = torch.from_numpy(resize_img)
        scale_x = float(width)/float(img.shape[1])
        scale_y = float(height)/float(img.shape[0])
        annots[i][:, 0] = annots[i][:, 0] * scale_x
        annots[i][:, 1] = annots[i][:, 1] * scale_y
        annots[i][:, 2] = annots[i][:, 2] * scale_x
        annots[i][:, 3] = annots[i][:, 3] * scale_y
        mean = np.array([[[0.485, 0.456, 0.406]]])
        std = np.array([[[0.229, 0.224, 0.225]]])
        padded_imgs[i, :, :, :] = torch.from_numpy(
            (resize_img.astype(np.float32) / 255.0 - mean) / std)
        im_info[i, :] = torch.from_numpy(
            np.array([height, width, scale_x, scale_y], dtype=float))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * 0

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0],
                                 :] = torch.from_numpy(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * 0

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    # return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
    return {'img': padded_imgs, 'annot': annot_padded, 'im_info': im_info}


class ResizerMultiScale(object):
    def __call__(self, sample, min_side=608, max_side=800):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        scales = [scale for scale in range(min_side, max_side, 32)]

        real_min_side = scales[random.randint(0, len(scales) - 1)]

        # rescale the image so the smallest side is min_side
        scale = real_min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side
            image = cv2.resize(
                image, (int(round((cols * scale))), int(round(rows * scale))))
        rows, cols, cns = image.shape

        pad_w = 0  # 32 - rows % 32
        pad_h = 0  # 32 - cols % 32

        new_image = np.zeros(
            (rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=640, max_side=1920):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        # smallest_side = min(rows, cols)
        #
        # # rescale the image so the smallest side is min_side
        # scale = min_side / smallest_side
        #
        # # check if the largest side is now greater than max_side, which can happen
        # # when images have a large aspect ratio
        # largest_side = max(rows, cols)
        #
        # if largest_side * scale > max_side:
        #     scale = max_side / largest_side
        #
        # # resize the image with the computed scale
        # #image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))), mode='constant')
        # image = cv2.resize(image, (int(round((cols * scale))), int(round(rows * scale)) ) )
        # rows, cols, cns = image.shape
        #
        # pad_w = 32 - rows % 32
        # pad_h = 32 - cols % 32
        #
        # new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_image[:rows, :cols, :] = image.astype(np.float32)
        #
        # annots[:, :4] *= scale

        return {'img': image, 'annot': torch.from_numpy(annots), 'scale': 1}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        image, annots = sample['img'], sample['annot']

        # self.vis_flip(image, annots, "/data/zhangcc/data/tmp/original.jpg")

        rdm = np.random.random()

        if rdm < 0.25:
            image, annots = self.image_flip_x(image, annots)  # h
        elif rdm < 0.5:
            image, annots = self.image_flip_y(image, annots)  # v
        elif rdm < 0.75:
            image, annots = self.image_flip_xy(image, annots)  # hv

        # self.vis_flip(image, annots, "/data/zhangcc/data/tmp/flip.jpg")

        sample = {'img': image, 'annot': annots}

        return sample

    def vis_flip(self, image, annots, image_path=None):
        import cv2

        image_cpy = image.copy()
        for annot in annots:
            x1, y1 = int(annot[0]), int(annot[1])
            x2, y2 = int(annot[2]), int(annot[3])
            cv2.rectangle(image_cpy, (x1, y1), (x2, y2), (255, 0, 255), 5)
            cv2.imwrite(image_path, image_cpy)

        return image_cpy

    def image_flip_x(self, image, annots):
        _, image_w, _ = image.shape

        image = image[:, ::-1, :]

        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()

        x_tmp = x1.copy()

        annots[:, 0] = image_w - 1 - x2
        annots[:, 2] = image_w - 1 - x_tmp

        return image, annots

    def image_flip_y(self, image, annots):
        image_h, _, _ = image.shape

        image = image[::-1, :, :]

        y1 = annots[:, 1].copy()
        y2 = annots[:, 3].copy()

        y_tmp = y1.copy()

        annots[:, 1] = image_h - 1 - y2
        annots[:, 3] = image_h - 1 - y_tmp

        return image, annots

    def image_flip_xy(self, image, annots):
        image, annots = self.image_flip_x(image, annots)
        image, annots = self.image_flip_y(image, annots)

        return image, annots


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(
            key=lambda x: self.data_source.image_aspect_ratio(x), reverse=True)

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = ('BACKGROUND',
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'
                            )
        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str)
                                if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
