import random

import numpy as np
import torch




# 基于分布的噪声
# 基于频率和振幅的噪声, BW
# 直线噪声
# 基于时间片段的噪声和基于实例的噪声
# 噪声的叠加方式： 乘，加


# 多通道 先选通道 再选噪声类型 也可能是不添加噪声
# 单通道 选定噪声类型 也可能是 不加噪声
from ecg_dataset.augmentation import Select_Win, StraightLine_1D, Gussian_Noisy_1D, Rayleign_Noisy_1D, Gamma_Noisy_1D, \
    Poisson_Noisy_1D, Exponential_Noisy_1D, Uniform_Noisy_1D, Am_Noisy_1D, Fm_Noisy_1D, Bw_Noisy_1D


class Raw:
    def __init__(self):
        pass

    def __call__(self, data):
        # print('Raw')
        return data


class Win:

    def __init__(self, min_win, seed=None):
        self.min_win = min_win
        self.seed = seed

    def __call__(self, data):
        # print('Select Window Size......')

        start, end = Select_Win(data, self.min_win, seed=self.seed)

        return start, end


class ChannelStraight_1D:
    def __init__(self, p, is_selectwin):
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, start, end):

        if random.random() < self.p:

            # print('Select Channel To Straight Line.....')
            return self.forward(data, start, end)
        else:
            return data

    def forward(self, data, start, end):
        trans = StraightLine_1D(data, self.is_select, start, end)
        return trans


class Gussian:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, start_index, end_index):
        # a = random.random()
        if random.random() < self.p:
            # print('Gussian ing')
            return self.forward(data, start_index, end_index)
        return data

    def forward(self, data, start_index, end_index):
        trans = Gussian_Noisy_1D(data, self.snr, self.is_select, start_index, end_index)
        return trans


class Rayleign:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Rayleign ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Rayleign_Noisy_1D(data, self.snr, self.is_select, win_start, win_end)
        return trans


class Gamma:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Gamma ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Gamma_Noisy_1D(data, self.snr, self.is_select, win_start, win_end)
        return trans


class Poisson:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Poisson ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Poisson_Noisy_1D(data, self.snr, self.is_select, win_start, win_end)
        return trans


class Exponential:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Exponential ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Exponential_Noisy_1D(data, self.snr, self.is_select, win_start, win_end)
        return trans


class Uniform:
    def __init__(self, snr, p, is_selectwin):
        self.snr = snr
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Uniform ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Uniform_Noisy_1D(data, self.snr, self.is_select, win_start, win_end)
        return trans


class Am:
    def __init__(self, fs, amplitude, p, is_selectwin):
        self.fs = fs
        self.amplitude = amplitude
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, start_index, end_index):
        if random.random() < self.p:
            # print('Am ing')
            return self.forward(data, start_index, end_index)
        return data

    def forward(self, data, start_index, end_index):
        trans = Am_Noisy_1D(data, self.fs, self.amplitude, self.is_select, start_index, end_index)
        return trans


class Fm:
    def __init__(self, fs, amplitude, p, is_selectwin):
        self.fs = fs
        self.amplitude = amplitude
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Fm ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Fm_Noisy_1D(data, self.fs, self.amplitude, self.is_select, win_start, win_end)
        return trans


class Bw:
    def __init__(self, fs, amplitude, p, is_selectwin):
        self.fs = fs
        self.amplitude = amplitude
        self.p = p
        self.is_select = is_selectwin

    def __call__(self, data, win_start, win_end):
        if random.random() < self.p:
            # print('Bw ing')
            return self.forward(data, win_start, win_end)
        return data

    def forward(self, data, win_start, win_end):
        trans = Bw_Noisy_1D(data, self.fs, self.amplitude, self.is_select, win_start, win_end)
        return trans


class ToTensor:
    '''
    Attributes
    ----------
    basic : convert numpy to PyTorch tensor

    Methods
    -------
    forward(img=input_image)
        Convert HWC OpenCV image into CHW PyTorch Tensor
    '''

    def __init__(self, basic=False):
        self.basic = basic

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv/numpy image

        Returns
        -------
        Torch tensor
            BGR -> RGB, [0, 255] -> [0, 1]
        '''
        ret = torch.from_numpy(img).type(torch.FloatTensor)
        return ret


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)

        return img


class Compose_Multi:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        for t in self.transforms:
            data = t(data)

        return data


class Compose_Select:
    def __init__(self, transforms, min_win):
        self.transforms = transforms
        self.compose_single = []
        self.compose_all = []
        self.min_win = min_win

    def __call__(self, data):
        return self.forward(data)

    def select_win(self, min_win, signal_length, seed):
        random.seed(seed)
        start_index = random.randint(0, signal_length - min_win)
        end_index = start_index + min_win

        return start_index, end_index

    def forward(self, data):
        random.seed(0)  # You may adjust the seed as needed
        self.compose_all = []
        start_index, end_index = self.select_win(self.min_win, data.shape[0], 0)

        for j in range(data.shape[0]):
            single = data[j]
            for t in self.transforms:
                single = t(single, start_index, end_index)
            self.compose_all.append(np.squeeze(single))

        self.compose_all = np.array(self.compose_all)
        return self.compose_all


class Compose_Add:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        add = data
        for t in self.transforms:
            noisy = t(data)
            add = add + noisy

        return add
