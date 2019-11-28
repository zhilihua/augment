"""
光学上的变换操作
"""
from __future__ import division
import numpy as np
import cv2

class ConvertColor:
    '''
    在RGB,HSV和灰度图之间进行转换。用的是cv2的api
    '''
    def __init__(self, current='RGB', to='HSV', keep_3ch=True):
        '''
        参数:
            current (str, optional): 当前的数据格式，仅仅是'GRB'和'HSV'中的一个。
            to (str, optional): 转换为的数据格式，3种模式都可以。
            keep_3ch (bool, optional): 仅仅与to==GRAY相关。
                如果为'True'，得到的灰度尺寸图像有3个通道。
        '''
        if not ((current in {'RGB', 'HSV'}) and (to in {'RGB', 'HSV', 'GRAY'})):
            raise NotImplementedError
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None):
        if self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'RGB' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        elif self.current == 'HSV' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif self.current == 'HSV' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertDataType:
    '''
    在uint8和float32之间转换为Numpy数组表示的图像。转换结果为numpy数据
    '''
    def __init__(self, to='uint8'):
        '''
        参数:
            to (string, optional): 转换为的数据格式。
        '''
        if not (to == 'uint8' or to == 'float32'):
            raise ValueError("`to` can be either of 'uint8' or 'float32'.")
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertTo3Channels:
    '''
    转换1通道和4通道数据到3通道。当数据为3通道的时候不做任何操作。
    在4通道图像的情况下，第四通道将被丢弃。
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        if labels is None:
            return image
        else:
            return image, labels

class Hue:
    '''
    改变HSV图像得色相。opencv中取值为H：（0-180），S：（0-255），V：（0-255）
    '''
    def __init__(self, delta):
        '''
        参数:
            delta (int): 闭合间隔“ [-180，180]”中的一个整数，用于确定色调变化，
            其中整数“ delta”的变化表示“ 2 * delta”度的变化。
        '''
        if not (-180 <= delta <= 180):raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels

class RandomHue:
    '''
    随机改变图像HSV中的H值。
    '''
    def __init__(self, max_delta=18, prob=0.5):
        '''
        参数:
            max_delta (int): 闭区间[0-180]之间的一个值，决定了最大绝对值H的改变。
            prob (float, optional): 进行H值变化的概率。
        '''
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Saturation:
    '''
    改变图像HSV中的S值。
    '''
    def __init__(self, factor):
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomSaturation:
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Brightness:
    '''
    改变RGB图像的明亮度。
    '''
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomBrightness:

    def __init__(self, lower=-84, upper=84, prob=0.5):

        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

#更改图像的对比度
class Contrast:
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): 大于零的浮点决定了对比度变化，
            其中小于1的值导致对比度降低，而大于1的值导致对比度更高。
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomContrast:

    def __init__(self, lower=0.5, upper=1.5, prob=0.5):

        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

#对图像进行gamma变换，达到像素缩减的功能
class Gamma:
    '''
    改变RGB图像的gamma值。
    '''
    def __init__(self, gamma):
        '''
        参数:
            gamma (float): 一个大于0的值，其决定gamma的改变。
        '''
        if gamma <= 0.0: raise ValueError("It must be `gamma > 0`.")
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma

        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, labels=None):
        image = cv2.LUT(image, self.table)
        if labels is None:
            return image
        else:
            return image, labels

class RandomGamma:

    def __init__(self, lower=0.25, upper=2.0, prob=0.5):

        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            gamma = np.random.uniform(self.lower, self.upper)
            change_gamma = Gamma(gamma=gamma)
            return change_gamma(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

#直方图均衡化
class HistogramEqualization:
    '''
    对HSV图像执行直方图均衡化。
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        if labels is None:
            return image
        else:
            return image, labels

class RandomHistogramEqualization:
    '''
    在HSV图像上随机执行直方图均衡化。 随机性仅是指是否进行均衡。
    '''
    def __init__(self, prob=0.5):
        self.prob = prob
        self.equalize = HistogramEqualization()

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.equalize(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class ChannelSwap:
    '''
    交换图像通道。
    '''
    def __init__(self, order):
        '''
        Arguments:
            order (tuple): 整数元组，用于定义通道交换后输入图像的所需通道顺序。
        '''
        self.order = order

    def __call__(self, image, labels=None):
        image = image[:,:,self.order]
        if labels is None:
            return image
        else:
            return image, labels

class RandomChannelSwap:

    def __init__(self, prob=0.5):

        self.prob = prob
        # 除原始顺序外，三个图像通道的所有可能排列。
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5)
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels