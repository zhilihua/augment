"""
进行数据增广的整理流程
"""
import numpy as np
import cv2
import inspect

from object_detection_2d_photometric import ConvertColor, ConvertDataType, ConvertTo3Channels, RandomBrightness, RandomContrast, RandomHue, RandomSaturation, RandomChannelSwap
from object_detection_2d_patch_sampling import PatchCoordinateGenerator, RandomPatch, RandomPatchInf
from object_detection_2d_geometric import ResizeRandomInterp, RandomFlip
from object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class RandomCrop:
    '''
    执行`batch_sampler`指令所定义的随机裁剪。
    '''

    def __init__(self, labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''

        self.labels_format = labels_format

        # 每次调用“ sample_space”定义的IoU下界之一。
        self.bound_generator = BoundGenerator(sample_space=((None, None),
                                                            (0.1, None),
                                                            (0.3, None),
                                                            (0.5, None),
                                                            (0.7, None),
                                                            (0.9, None)),
                                              weights=None)

        # 生成候选补丁的坐标，以使补丁的高度和宽度在相应图像的高度和宽度的0.3到1.0之间，
        # 补丁的纵横比在0.5到2.0之间。
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=0.3,
                                                              max_scale=1.0,
                                                              scale_uniformly=False,
                                                              min_aspect_ratio = 0.5,
                                                              max_aspect_ratio = 2.0)

        # 筛选出中心点不在所选补丁内的框。
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion='center_point',
                                    labels_format=self.labels_format)

        # 确定给定补丁是否被视为有效补丁。 如果至少一个真实边界框（n_boxes_min == 1）与
        # 该补丁的IoU重叠满足“ bound_generator”定义的要求，则将该补丁定义为有效。
        self.image_validator = ImageValidator(overlap_criterion='iou',
                                              n_boxes_min=1,
                                              labels_format=self.labels_format,
                                              border_pixels='half')

        # 根据以上对象中设置的参数执行裁剪。 一直运行到找到有效补丁或原始输入图像未更改返回为止。
        # 最多运行50次试验，以为每个新采样的IoU阈值找到有效的补丁。
        # 每50次试验，原图像的返回概率为（1-prob）= 0.143。
        self.random_crop = RandomPatchInf(patch_coord_generator=self.patch_coord_generator,
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          bound_generator=self.bound_generator,
                                          n_trials_max=50,
                                          clip_boxes=True,
                                          prob=0.857,
                                          labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        return self.random_crop(image, labels, return_inverter)

class Expand:
    '''
    执行“ train_transform_param”指令所定义的随机图像扩展。
    '''

    def __init__(self, background=(123, 117, 104),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            background (list/tuple, optional): 一个三元组，指定转换图像的背景像素的RGB颜色值。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''

        self.labels_format = labels_format

        # 在两个空间维度上为输入图像大小的1.0到4.0倍之间的补丁生成坐标。
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=1.0,
                                                              max_scale=4.0,
                                                              scale_uniformly=True)

        # 根据上面设置的参数，以0.5的概率将输入图像随机放置在填充有平均颜色值的画布上。
        # 以0.5的概率返回未更改的输入图像。
        self.expand = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                  box_filter=None,
                                  image_validator=None,
                                  n_trials_max=1,
                                  clip_boxes=False,
                                  prob=0.5,
                                  background=background,
                                  labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.expand.labels_format = self.labels_format
        return self.expand(image, labels, return_inverter)

class PhotometricDistortions:
    '''
    执行“ train_transform_param”指令定义的光学变换。
    '''

    def __init__(self):

        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.0)

        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_channel_swap]

        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.convert_to_float32,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.random_channel_swap]

    def __call__(self, image, labels):
        if np.random.choice(2):

            for transform in self.sequence1:
                image, labels = transform(image, labels)
            return image, labels
        else:

            for transform in self.sequence2:
                image, labels = transform(image, labels)
            return image, labels

class DataAugmentation:
    '''
    实现的数据增强管道。
    '''

    def __init__(self,
                 img_height=300,
                 img_width=300,
                 background=(123, 117, 104),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            background (list/tuple, optional): 一个三元组，指定转换图像的背景像素的RGB颜色值。
            labels_format (dict, optional): 一个字典，它定义图像标签的最后一个轴中的哪个索引包含哪个边界框坐标。
        '''

        self.labels_format = labels_format

        self.photometric_distortions = PhotometricDistortions()
        self.expand = Expand(background=background, labels_format=self.labels_format)
        self.random_crop = RandomCrop(labels_format=self.labels_format)
        self.random_flip = RandomFlip(dim='horizontal', prob=0.5, labels_format=self.labels_format)

        # 此框过滤器可确保调整大小后的图像不包含任何退化的框。 调整图像大小可能会导致盒子变小。
        # 对于已经很小的盒子，可能导致盒子的高度和/或宽度为零，这显然是我们不允许的。
        self.box_filter = BoxFilter(check_overlap=False,
                                    check_min_area=False,
                                    check_degenerate=True,
                                    labels_format=self.labels_format)

        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4],
                                         box_filter=self.box_filter,
                                         labels_format=self.labels_format)

        self.sequence = [self.photometric_distortions,
                         self.expand,
                         self.random_crop,
                         self.random_flip,
                         self.resize]

    def __call__(self, image, labels, return_inverter=False):
        self.expand.labels_format = self.labels_format
        self.random_crop.labels_format = self.labels_format
        self.random_flip.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        inverters = []

        for transform in self.sequence:
            if return_inverter and ('return_inverter' in inspect.signature(transform).parameters):
                image, labels, inverter = transform(image, labels, return_inverter=True)
                inverters.append(inverter)
            else:
                image, labels = transform(image, labels)

        if return_inverter:
            return image, labels, inverters[::-1]
        else:
            return image, labels

if __name__ == '__main__':
    #展示效果
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    labels = [[0, 174, 101, 349, 351]]
    img = np.array(Image.open('dataset/2007_000027.jpg'))

    #进行数据增广


    plt.imshow(img)
    rect = plt.Rectangle((labels[0][1], labels[0][2]), labels[0][3]-labels[0][1],
                         labels[0][4]-labels[0][2], color='r', fill=False, linewidth=2)  # 左下起点，长，宽，颜色
    #画矩形框
    plt.gca().add_patch(rect)

    plt.show()