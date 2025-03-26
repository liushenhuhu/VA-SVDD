import torch
import torch.utils.data as data
import pywt
import numpy as np

from torch.utils.data import DataLoader
from ecg_dataset import transform
from ecg_dataset.load_all_data import load_vfdb, load_mitbih, load_cudb
from scipy.fftpack import fft
from scipy import signal

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
def to_frequency(signal):
    #傅里叶
    # signal = fft(signal,axis=0)
    # signal = np.abs(signal)
    #小波
    signal = pywt.swt(signal, 'db1', level=2)[0][0]

    return signal

class TransformDataset_SelectChannel_2:
    def __init__(self, x, y, snr, fs, amplitude, p, min_winsize, is_selectwin, line_num):
        x_length =x.shape[-1]  # n,12,2500



        straight = transform.ChannelStraight_1D(p=p, is_selectwin=is_selectwin)
        gussian = transform.Gussian(snr=snr, p=p, is_selectwin=is_selectwin)
        gamma = transform.Gamma(snr=snr, p=p, is_selectwin=is_selectwin)
        rayleign = transform.Rayleign(snr=snr, p=p, is_selectwin=is_selectwin)
        exponential = transform.Exponential(snr=snr, p=p, is_selectwin=is_selectwin)
        poisson = transform.Poisson(snr=snr, p=p, is_selectwin=is_selectwin)
        uniform = transform.Uniform(snr=snr, p=p, is_selectwin=is_selectwin)
        am = transform.Am(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)
        fm = transform.Fm(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)
        bw = transform.Bw(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)
        transforms_list = {
            'gussian': gussian,
            'gamma': gamma,
            'rayleign': rayleign,
            'exponential': exponential,
            'poisson': poisson,
            'uniform': uniform,
            'fm': fm,
            'am': am,
            'bw': bw,
            'line': straight
        }
        trans = transform.Compose_Select(transforms_list.values(), min_winsize) #将多个变换组合在一起，并顺序应用它们

        # 原数据
        x = np.asarray(x, dtype=np.float32)
        # 伪噪声数据
        noisy = trans(x.reshape(-1, x_length))
        noisy = noisy.reshape(-1, 1, x_length)
        # 伪VF数据
        # time = np.linspace(0, 10, x_length)  # 时间范围从0到1，共2500个数据点
        # modulated_signal = frequency_modulate(time, x, 2, 0.1) #方法1
        modulated_signal = frequency_modulate2(x, x_length, 2, 0.1, 0.5, 0.5, (0.2, 0.7)) #方法2

        # 原数据-频域数据
        x_F = to_frequency(x)
        # 伪噪声数据-频域数据
        noisy_F =to_frequency(noisy)
        # 伪VF数据-频域数据
        modulated_signal_F = to_frequency(modulated_signal)

        # 原数据标签
        y = np.asarray(y, dtype=np.int64)
        # 伪噪声标签
        y_noisy = np.full(len(y), 1)
        # 伪VF数据标签
        y_modulated_signal = np.full(len(y), 2)

        self.x = torch.Tensor(x)
        self.noisy = torch.Tensor(noisy)
        self.modulated_signal = torch.Tensor(modulated_signal)

        self.x_F = torch.Tensor(x_F)
        self.noisy_F = torch.Tensor(noisy_F)
        self.modulated_signal_F = torch.Tensor(modulated_signal_F)

        self.y = torch.Tensor(y)
        self.y_noisy = torch.Tensor(y_noisy)
        self.y_modulated_signal = torch.Tensor(y_modulated_signal)

    def __getitem__(self, index):
        return self.x[index],self.noisy[index],self.modulated_signal[index], self.x_F[index],self.noisy_F[index],self.modulated_signal_F[index], self.y[index],self.y_noisy[index],self.y_modulated_signal[index]

    def __len__(self):
        return self.x.shape[0]


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






