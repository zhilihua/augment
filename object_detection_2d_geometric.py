from __future__ import division
import numpy as np
import cv2
from object_detection_2d_image_boxes_validation_utils import BoxFilter

class Resize:
    '''
    将图像调整为指定的高度和宽度（以像素为单位）。
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            interpolation_mode (int, optional): 表示有效的OpenCV插值模式的整数。
            box_filter (BoxFilter, optional): 仅在给出真值边界框的情况下才相关。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

class ResizeRandomInterp:
    '''
    使用随机选择的插值模式将图像调整为特定的高度和宽度（以像素为单位）。
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            interpolation_mode (int, optional): 表示有效的OpenCV插值模式的整数。
            box_filter (BoxFilter, optional): 仅在给出真值边界框的情况下才相关。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)

class Flip:
    '''
    水平或垂直翻转图像。
    '''
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): 可以是“水平”和“垂直”之一。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels

class RandomFlip:
    '''
    水平或垂直随机翻转图像。 随机性仅指图像是否被翻转。
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): 可以是“水平”和“垂直”之一。
            prob (float, optional): 执行此操作的概率`(1 - prob)`。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels