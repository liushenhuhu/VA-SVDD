import torch
import torch.utils.data as data
import pywt
import numpy as np

from torch.utils.data import DataLoader
from ecg_dataset import transform
from ecg_dataset.load_all_data import load_vfdb, load_mitbih, load_cudb
from scipy.fftpack import fft
from scipy import signal
from ecg_dataset.ECG_dataloader import TransformDataset_SelectChannel_2
def frequency_modulate2(signal,x_length, frequency_base,
                        fm_modulation_index, am_modulation_index, am_frequency_modulation,
                        attenuation_factor_range=(0.2, 0.7)):

    # 生成一个频率随信号变化的频率信号（FM调制）
    time = np.linspace(0, 10, x_length)  # 时间范围从0到10秒，共2500个数据点（250Hz采样率）

    modulated_frequency = frequency_base + fm_modulation_index * signal
    modulated_signal_fm = np.cos(2 * np.pi * modulated_frequency * time)

    # 生成一个调幅信号（AM调制）
    modulated_amplitude = (1 + am_modulation_index * np.sin(2 * np.pi * am_frequency_modulation * time))
    modulated_signal_am = modulated_signal_fm * modulated_amplitude

    # 生成一个随机衰减因子
    attenuation_factor = np.random.uniform(*attenuation_factor_range)

    # 应用随机衰减因子
    vfake_signal = modulated_signal_am * attenuation_factor

    return vfake_signal


def get_dataloader(opt):
    global train_dataset, val_dataset, test_dataset

    train_data1, train_label1, val_data, val_label, test_data, test_label = load_vfdb(opt)

    train_data2, train_label2, val_data, val_label, test_data, test_label = load_mitbih(opt)
    # train_data2 = signal.resample(train_data2, 2500)

    train_data3, train_label3, val_data, val_label, test_data, test_label = load_cudb(opt)

    normal_data = np.concatenate((train_data1, train_data3), axis=0)
    normal_label = np.concatenate((train_label1, train_label3), axis=0)


    len_normal = len(normal_data)
    train_X = normal_data[:len_normal*2//3]
    train_Y = normal_label[:len_normal*2//3]
    val_X = normal_data[len_normal*2//3:]
    val_Y = normal_label[len_normal*2//3:]

    print("[INFO] Train: normal={}".format(train_X.shape), )
    print("[INFO] Val: normal={}".format(val_X.shape), )

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]


    train_dataset = TransformDataset_SelectChannel_2(train_X, train_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
    val_dataset = TransformDataset_SelectChannel_2(val_X, val_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)

    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length






