from .base_dataset import BaseDataset
from .registry import DATASETS
from . import transforms, utils

import os
from lxml import etree
import re
import cv2
import numpy as np

@DATASETS.register_module
class PascalDataset(BaseDataset):

    def __init__(self,
                 dataset_dir,
                 subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 preproc_mode='caffe',
                 scale=(1024, 800),
                 train=False,
                 debug=False):
        
        if subset not in ['train', 'val']:
            raise AssertionError('subset must be "train" or "val".')

        self.dataset_dir = dataset_dir
        self.files = []
        with open('{}/ImageSets/Main/{}.txt'.format(self.dataset_dir, subset), 'r') as f:
            for line in f:
                self.files.append(line[:-1])
        
        self._build_category_ids()
        self.classes = len(self.category_ids)

        # doesn't actually filter any images, but gives an example of what filtering might look like
        self.files = self._filter_images()

        if debug:
            self.files = self.files[:50]
        
        self.flip_ratio = flip_ratio

        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'
        
        self.img_transform = transforms.ImageTransform(scale, mean, std,
                                                       pad_mode)
        self.bbox_transform = transforms.BboxTransform()
        self.train = train
        self.preproc_mode = preproc_mode
    
    def _build_category_ids(self):
        files = os.listdir('{}/ImageSets/Main'.format(self.dataset_dir))
        categories = set()
        for file_name in files:
            match = re.match('[a-z]+_', file_name)
            if match:
                category = match.group(0)[:-1]
                categories.add(category)
        categories = sorted(categories)
        self.category_ids = {}
        for i, category in enumerate(categories):
            self.category_ids[category] = i + 1

    def _parse_xml(self, file_path):
        tree = etree.parse('{}/Annotations/{}.xml'.format(self.dataset_dir, file_path))
        root = tree.getroot()
        bboxes = []
        categories = []
        for child in root:
            if child.tag == 'size':
                for val in child:
                    if val.tag == 'width':
                        width = int(val.text)
                    if val.tag == 'height':
                        height = int(val.text)
            if child.tag == 'object':
                category = child[0].text
                for grandchild in child:
                    if grandchild.tag == 'bndbox':
                        area = [0, 0, 0, 0]
                        for val in grandchild:
                            if val.tag == 'ymin':
                                area[0] = int(val.text)
                            if val.tag == 'xmin':
                                area[1] = int(val.text)
                            if val.tag == 'ymax':
                                area[2] = int(val.text)
                            if val.tag == 'xmax':
                                area[3] = int(val.text)
                        if area[2] - area[0] < 1 or area[3] - area[1] < 1:
                            continue
                        categories.append(self.category_ids[category])
                        bboxes.append(area)
        return width, height, categories, bboxes

    def _filter_images(self, min_size=32):
        new_files = []
        for f in self.files:
            width, height, categories, bboxes = self._parse_xml(f)
            if min(width, height) < min_size or not bboxes:
                continue
            new_files.append(f)
        return new_files

    def get_ann_info(self, index):
        file_path = self.files[index]
        _, _, labels, bboxes = self._parse_xml(file_path)
        ann = dict(bboxes=np.array(bboxes),
                   labels=np.array(labels),
                   bboxes_ignore=np.zeros((0, 4), dtype=np.float32))
        return ann

    def num_classes(self):
        return self.classes
    
    def get_labels(self):
        return list(self.category_ids.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        bgr_img = cv2.imread('{}/JPEGImages/{}.jpg'.format(self.dataset_dir, file_path)).astype(np.float32)

        if self.preproc_mode == 'tf':
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = img/127.0 - 1.0
        elif self.preproc_mode == 'caffe':
            pixel_means = [103.939, 116.779, 123.68]
            channels = cv2.split(bgr_img)
            for i in range(3):
                channels[i] -= pixel_means[i]
            img = cv2.merge(channels)
        else:
            raise NotImplementedError

        orig_shape = img.shape 

        _, _, labels, bboxes = self._parse_xml(file_path)
        labels = np.array(labels)
        bboxes = np.array(bboxes)

        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        img, img_shape, scale_factor = self.img_transform(img, flip)

        pad_shape = img.shape

        bboxes, labels = self.bbox_transform(bboxes, labels, img_shape,
                                             scale_factor, flip)

        img_meta_dict = dict({
            'ori_shape': orig_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })

        img_meta = utils.compose_image_meta(img_meta_dict)
        if self.train:
            return img, img_meta, bboxes, labels
        return img, img_meta
    
    def _verify_output(self):
        item = self.__getitem__(0)
        assert len(item) == 4
        assert item[0].shape == (1024, 1024, 3)
        assert len(item[1]) == 11
        assert len(item[2]) == len(item[3])