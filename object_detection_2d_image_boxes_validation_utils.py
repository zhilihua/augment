from __future__ import division
import numpy as np
from box_utils import iou

class BoundGenerator:
    '''
        生成一对浮点值，它们代表给定样本空间的上下限。
    '''

    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):
        '''
        Arguments:
            sample_space (list or tuple): 可取的上下限的列表。
            weights (list or tuple, optional): 每个值被取到的概率。
        '''

        if (not (weights is None)) and len(weights) != len(sample_space):
            raise ValueError(
                "`weights` must either be `None` for uniform distribution or have the same length as `sample_space`.")

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError("All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None: bound_pair[0] = 0.0
            if bound_pair[1] is None: bound_pair[1] = 1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError(
                    "For all sample space elements, the lower bound cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0 / self.sample_space_size] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]

class BoxFilter:
    '''
        返回关于定义的条件有效的所有边界框。
    '''

    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Arguments:
            check_overlap (bool, optional): 是否强制执行“ overlap_criterion”和“ overlap_bounds”定义的重叠要求。
            check_min_area (bool, optional): 是否强制执行被要求的最小面积。
            check_degenerate (bool, optional): 是否检查并删除退化的边界框。
            overlap_criterion (str, optional): 可以是“ center_point”，“ iou”或“ area”。
                确定哪些框相对于给定图像有效。
            overlap_bounds (list or BoundGenerator, optional): 仅在“ overlap_criterion”为“ area”或“ iou”时才相关。
            min_area (int, optional): 仅在“ check_min_area”为“ True”时才相关。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
            border_pixels (str, optional): 如何处理边界框的边框像素。 可以是“ include”，“ exclude”或“ half”。
        '''
        if not isinstance(overlap_bounds, (list, tuple, BoundGenerator)):
            raise ValueError("`overlap_bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(overlap_bounds, (list, tuple)) and (overlap_bounds[0] > overlap_bounds[1]):
            raise ValueError("The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError("`overlap_criterion` must be one of 'iou', 'area', or 'center_point'.")
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        '''
        Arguments:
            labels (array): 要过滤的标签。
            image_height (int): 仅在“ check_overlap == True”时相关。 与框坐标进行比较的图像高度（以像素为单位）。
            image_width (int): `check_overlap == True`。 与框坐标进行比较的图像宽度（以像素为单位）。
        Returns:
            包含所有有效框标签的数组。
        '''

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # 在此处记录通过所有检查的框。
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:
            non_degenerate = (labels[:, xmax] > labels[:, xmin]) * (labels[:, ymax] > labels[:, ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:
            min_area_met = (labels[:, xmax] - labels[:, xmin]) * (labels[:, ymax] - labels[:, ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:

            # 获取上下限。
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # 计算哪些框有效。

            if self.overlap_criterion == 'iou':
                # 计算补丁坐标。
                image_coords = np.array([0, 0, image_width, image_height])
                # 计算补丁和所有真实包围框之间的IoU。
                image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]], coords='corners',
                                      mode='element-wise', border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou > lower) * (image_boxes_iou <= upper)

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    d = 1
                elif self.border_pixels == 'exclude':
                    d = -1
                # 计算包围框的面积。
                box_areas = (labels[:, xmax] - labels[:, xmin] + d) * (labels[:, ymax] - labels[:, ymin] + d)
                # 计算补丁和所有真值框之间的交集区域。
                clipped_boxes = np.copy(labels)
                clipped_boxes[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=image_height - 1)
                clipped_boxes[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=image_width - 1)
                intersection_areas = (clipped_boxes[:, xmax] - clipped_boxes[:, xmin] + d) * (
                            clipped_boxes[:, ymax] - clipped_boxes[:,
                                                     ymin] + d)
                # 检查哪些框满足重叠要求。
                if lower == 0.0:
                    mask_lower = intersection_areas > lower * box_areas
                else:
                    mask_lower = intersection_areas >= lower * box_areas
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # 计算框的中心点。
                cy = (labels[:, ymin] + labels[:, ymax]) / 2
                cx = (labels[:, xmin] + labels[:, xmax]) / 2
                # 检查哪些方框中有裁剪的补丁程序中的中心点，删除那些没有的中心点。
                requirements_met *= (cy >= 0.0) * (cy <= image_height - 1) * (cx >= 0.0) * (cx <= image_width - 1)

        return labels[requirements_met]

class ImageValidator:
    '''
        如果给定的最小数量的边界框满足给定的重叠要求且具有给定的高度和宽度的图像，则返回“ True”。
    '''

    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 n_boxes_min=1,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Arguments:
            overlap_criterion (str, optional): 可以是“ center_point”，“ iou”或“ area”。
                确定哪些框相对于给定图像有效。
            bounds (list or BoundGenerator, optional): 仅在“ overlap_criterion”为“ area”或“ iou”时才相关。
                确定“ overlap_criterion”的上下限。
            n_boxes_min (int or str, optional): 保留最少的包围框个数，非负整数或字符串“ all”。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
            border_pixels (str, optional): 如何处理边界框的边框像素。
        '''
        if not ((isinstance(n_boxes_min, int) and n_boxes_min > 0) or n_boxes_min == 'all'):
            raise ValueError("`n_boxes_min` must be a positive integer or 'all'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format
        self.border_pixels = border_pixels
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds,
                                    labels_format=self.labels_format,
                                    border_pixels=self.border_pixels)

    def __call__(self,
                 labels,
                 image_height,
                 image_width):
        '''
        Arguments:
            labels (array): 要测试的标签。
            image_height (int): 与框坐标进行比较的图像高度。
            image_width (int): 与框坐标进行比较的图像宽度。
        Returns:
            一个布尔值，指示相对于给定的边界框，给定的高度和宽度的图像是否有效。
        '''

        self.box_filter.overlap_bounds = self.bounds
        self.box_filter.labels_format = self.labels_format

        # 获取所有符合重叠要求的包围框。
        valid_labels = self.box_filter(labels=labels,
                                       image_height=image_height,
                                       image_width=image_width)

        # 检查是否有足够的包围框满足要求。
        if isinstance(self.n_boxes_min, int):
            # 如果至少`self.n_boxes_min`真实框满足要求，则该图像有效。
            if len(valid_labels) >= self.n_boxes_min:
                return True
            else:
                return False
        elif self.n_boxes_min == 'all':
            # 如果所有真实框均符合要求，则该图像有效。
            if len(valid_labels) == len(labels):
                return True
            else:
                return False