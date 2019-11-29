from __future__ import division
import numpy as np
from object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class PatchCoordinateGenerator:
    '''
    生成满足指定要求的随机补丁坐标。
    '''

    def __init__(self,
                 img_height=None,
                 img_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        '''
        Arguments:
            img_height (int): 图片的高度。
            img_width (int): 图片的宽度。
            must_match (str, optional): 匹配格式（'h_w', 'h_ar', 和 'w_ar'）。
                指定高度，宽度和纵横比这三个量中的哪两个确定所生成补丁的形状。
                各自的第三个量将从其他两个量计算得出。
            min_scale (float, optional): 图像中各个维度数值的最小值。
            max_scale (float, optional): 图像中各个维度数值的最大值。
            scale_uniformly (bool, optional): 如果为True，并且如果`must_match == 'h_w'`，
            则补丁高度和宽度将统一缩放，否则，它们将独立缩放。
            min_aspect_ratio (float, optional): 确定所生成补丁的最小纵横比。
            max_aspect_ratio (float, optional): 确定所生成补丁的最大纵横比。
            patch_ymin (int, optional): “无”或所生成补丁的左上角的垂直坐标。 如果不是“无”，
            则补丁沿垂直轴的位置是固定的。 如果此值为“无”，则将随机选择生成的补丁的垂直位置，
            以使补丁和图像沿垂直方向的重叠始终最大。
            patch_xmin (int, optional): 原理同patch_xmin
            patch_height (int, optional): 空或者固定高度。
            patch_width (int, optional): 空或者固定宽度。
            patch_aspect_ratio (float, optional): 空或者固定纵横比。
        '''

        if not (must_match in {'h_w', 'h_ar', 'w_ar'}):
            raise ValueError("`must_match` must be either of 'h_w', 'h_ar' and 'w_ar'.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError("It must be `min_aspect_ratio < max_aspect_ratio`.")
        if scale_uniformly and not ((patch_height is None) and (patch_width is None)):
            raise ValueError("If `scale_uniformly == True`, `patch_height` and `patch_width` must both be `None`.")
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        '''
        返回:
            四维元组 `(ymin, xmin, height, width)`
        '''

        # 得到补丁的高度和宽度。
        if self.must_match == 'h_w': #纵横比依赖变量
            if not self.scale_uniformly:
                # 得到高度。
                if self.patch_height is None:
                    patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
                else:
                    patch_height = self.patch_height
                # 得到宽度。
                if self.patch_width is None:
                    patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
                else:
                    patch_width = self.patch_width
            else:
                scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
                patch_height = int(scaling_factor * self.img_height)
                patch_width = int(scaling_factor * self.img_width)

        elif self.must_match == 'h_ar': # 宽度依赖变量。
            # 得到高度。
            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
            else:
                patch_height = self.patch_height
            # 得到纵横比例。
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # 得到宽度。
            patch_width = int(patch_height * patch_aspect_ratio)

        elif self.must_match == 'w_ar': # 高度依赖变量。
            # 得到宽度。
            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
            else:
                patch_width = self.patch_width
            # 得到纵横比例。
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # 得到高度。
            patch_height = int(patch_width / patch_aspect_ratio)

        # 得到补丁的左上角坐标。

        if self.patch_ymin is None:
            # 计算有多少空间，当我们沿着垂直方向放补丁时候。
            # 一个负数意味着我们想采样一个补丁，其大于原始图像。
            # 在垂直维度上，在这种情况下，将放置补丁，使其完全包含垂直尺寸的图像。
            y_range = self.img_height - patch_height
            # 从可能的位置中为样本位置选择一个随机的左上角。
            if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1)
            else: patch_ymin = np.random.randint(y_range, 1)
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:
            #同理patch_xmin
            x_range = self.img_width - patch_width
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1)
            else: patch_xmin = np.random.randint(x_range, 1)
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)

class CropPad:
    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_ymin (int, optional): 补丁左上角的y值。
            patch_xmin (int, optional): 补丁左上角的x值。
            patch_height (int): 补丁的高度。
            patch_width (int): 补丁的宽度。
            clip_boxes (bool, optional): 仅仅涉及如果真值包围框给定情况下。
                如果为`True`，任意真值包围框将被裁剪到采样补丁的内部。
            box_filter (BoxFilter, optional): 仅仅涉及如果真值包围框给定情况下。
                一个BoxFilter对象，用于过滤转换后不满足给定条件的边界框。如果是`None`,
                包围框的验证不进行检测。
            background (list/tuple, optional): 一个三元组，指定缩放图像的潜在背景像素的RGB颜色值。
             在单通道的图片中，第一个值将被使用。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError("The given patch doesn't overlap with the input image.")

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # 补丁相对于图像坐标系的左上角：
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        # 创建一个画布，其大小与我们最终想要的补丁相同。
        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3), dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width), dtype=np.uint8)
            canvas[:, :] = self.background[0]

        # 执行pad
        if patch_ymin < 0 and patch_xmin < 0: # 在上面和左边进行pad
            image_crop_height = min(img_height, self.patch_height + patch_ymin)
            image_crop_width = min(img_width, self.patch_width + patch_xmin)
            canvas[-patch_ymin:-patch_ymin + image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[:image_crop_height, :image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0: # 顶部pad，左部crop。
            image_crop_height = min(img_height, self.patch_height + patch_ymin)
            image_crop_width = min(self.patch_width, img_width - patch_xmin)
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :image_crop_width] = image[:image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0: # 顶部crop，左部pad。
            image_crop_height = min(self.patch_height, img_height - patch_ymin)
            image_crop_width = min(img_width, self.patch_width + patch_xmin)
            canvas[:image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, :image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0: # 全部进行crop。
            image_crop_height = min(self.patch_height, img_height - patch_ymin)
            image_crop_width = min(self.patch_width, img_width - patch_xmin)
            canvas[:image_crop_height, :image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] += patch_ymin
                labels[:, [xmin+1, xmax+1]] += patch_xmin
                return labels

        if not (labels is None):

            # 转换包围框的坐标到补丁的坐标系统。
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            # 对这个补丁，计算所有有效包围框。
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=self.patch_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=self.patch_width-1)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image

class RandomPatch:

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 prob=1.0,
                 background=(0, 0, 0),
                 can_fail=False,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): 一个`PatchCoordinateGenerator`对象用于生成补丁的
            位置和尺寸，补丁采样于输入图片
            box_filter (BoxFilter, optional): 仅仅涉及如果真值包围框给定情况下。
                一个BoxFilter对象，用于过滤转换后不满足给定条件的边界框。如果是`None`,
                包围框的验证不进行检测。
            image_validator (ImageValidator, optional): 仅仅涉及如果真值包围框给定情况下。
                一个`ImageValidator`对象用于决定是否一个采样补丁是有效的。如果是`None`，所有的输出都是有效的。
            n_trials_max (int, optional): 仅仅涉及如果真值包围框给定情况下。
                决定采样一个有效补丁的最大实验次数。如果没有有效补丁能够被采样到在`n_trials_max`次内，
                返回一个None。
            clip_boxes (bool, optional): 仅仅涉及如果真值包围框给定情况下。
                如果为True，
                如果 `True`, 任何真值包围框都将被裁剪为完全位于采样补丁中。
            prob (float, optional): `(1 - prob)` 决定随机采样的概率。
            background (list/tuple, optional): 一个三元组，指定缩放图像的潜在背景像素的RGB颜色值。
             在单通道的图片中，第一个值将被使用。
            can_fail (bool, optional): 如果为True，则在n_trials_max试用之后找不到有效补丁时，将返回None。
            如果为False，在这种情况下将返回未更改的输入图像。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''
        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.can_fail = can_fail
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]
            self.patch_coord_generator.img_height = img_height
            self.patch_coord_generator.img_width = img_width

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # 覆盖预设标签格式。
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.sample_patch.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # 生成补丁坐标。
                patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                self.sample_patch.patch_ymin = patch_ymin
                self.sample_patch.patch_xmin = patch_xmin
                self.sample_patch.patch_height = patch_height
                self.sample_patch.patch_width = patch_width

                if (labels is None) or (self.image_validator is None):
                    # 我们没有任何框，或者如果有，我们将接受任何有效的结果。
                    return self.sample_patch(image, labels, return_inverter)
                else:
                    # 转换包围框坐标到补丁的坐标系中。
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] -= patch_ymin
                    new_labels[:, [xmin, xmax]] -= patch_xmin
                    # 检查是否补丁是有效的。
                    if self.image_validator(labels=new_labels,
                                            image_height=patch_height,
                                            image_width=patch_width):
                        return self.sample_patch(image, labels, return_inverter)

            # 如果我们不能够采样到一个有效的补丁...，这一部分没有
            if self.can_fail:
                # ...返回 `None`.
                if labels is None:
                    if return_inverter:
                        return None, None
                    else:
                        return None
                else:
                    if return_inverter:
                        return None, None, None
                    else:
                        return None, None
            else:
                # ...返回不变的输入图像
                if labels is None:
                    if return_inverter:
                        return image, None
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, None
                    else:
                        return image, labels

        else:
            if return_inverter:
                def inverter(labels):
                    return labels

            if labels is None:
                if return_inverter:
                    return image, inverter
                else:
                    return image
            else:
                if return_inverter:
                    return image, labels, inverter
                else:
                    return image, labels


class RandomPatchInf:

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): 一个`PatchCoordinateGenerator`对象用于生成补丁的
            位置和尺寸，补丁采样于输入图片
            box_filter (BoxFilter, optional): 仅仅涉及如果真值包围框给定情况下。
                一个BoxFilter对象，用于过滤转换后不满足给定条件的边界框。如果是`None`,
                包围框的验证不进行检测。
            image_validator (ImageValidator, optional): 仅仅涉及如果真值包围框给定情况下。
                一个`ImageValidator`对象用于决定是否一个采样补丁是有效的。如果是`None`，所有的输出都是有效的。
            bound_generator (BoundGenerator, optional): 一个“ BoundGenerator”对象，为补丁验证器生成上限和下限值。
             每进行一次n_trials_max次试验，都会生成一对新的上下限，直到找到有效的补丁或返回原始图像为止。
              此绑定生成器将覆盖补丁验证器的绑定生成器。
            n_trials_max (int, optional): 仅仅涉及如果真值包围框给定情况下。
                决定采样一个有效补丁的最大实验次数。如果没有有效补丁能够被采样到在`n_trials_max`次内，
                返回一个None。
            clip_boxes (bool, optional): 仅仅涉及如果真值包围框给定情况下。
                如果为True，
                如果 `True`, 任何真值包围框都将被裁剪为完全位于采样补丁中。
            prob (float, optional): `(1 - prob)` 决定随机采样的概率。
            background (list/tuple, optional): 一个三元组，指定缩放图像的潜在背景像素的RGB颜色值。
             在单通道的图片中，第一个值将被使用。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''

        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        if not (isinstance(bound_generator, BoundGenerator) or bound_generator is None):
            raise ValueError("`bound_generator` must be either `None` or a `BoundGenerator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]
        self.patch_coord_generator.img_height = img_height
        self.patch_coord_generator.img_width = img_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # 覆盖预设标签格式。
        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True: # 持续运行知道我们发现一个有效的补丁或者返回原始图像。

            p = np.random.uniform(0,1)
            if p >= (1.0-self.prob):

                # 如果我们有绑定生成器，请为补丁验证器选择下限和上限。
                if not ((self.image_validator is None) or (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                # 最多使用`self.n_trials_max`尝试找到符合我们要求的裁剪区域。
                for _ in range(max(1, self.n_trials_max)):

                    # 生成补丁坐标。
                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    # 检查生成的补丁是否符合长宽比要求。
                    aspect_ratio = patch_width / patch_height
                    if not (self.patch_coord_generator.min_aspect_ratio <= aspect_ratio <= self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):
                        # 我们没有任何框，或者如果有，我们将接受任何有效的结果。
                        return self.sample_patch(image, labels, return_inverter)
                    else:
                        # 将框坐标转换为补丁的坐标系。
                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin
                        # 检查补丁是否包含我们要求的最少包围框数。
                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels, return_inverter)
            else:
                if return_inverter:
                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels