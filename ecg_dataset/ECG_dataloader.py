import torch
import torch.utils.data as data
import pywt
import numpy as np
from scipy.signal import welch

from torch.utils.data import DataLoader
from ecg_dataset import transform
from ecg_dataset.load_all_data import load_vfdb, load_mitbih, load_cudb
from scipy.fftpack import fft
class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.Tensor(X)
        self.X_f = to_frequency(X)
        self.Y = torch.Tensor(Y)
    def __getitem__(self, index):


        return self.X[index], self.X[index], self.Y[index], self.X[index],self.X[index],self.X[index],self.X[index]
    def __len__(self):
        return self.X.size(0)

class RawDataset2(data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.Tensor(X)
        self.X_f = to_frequency(X)
        self.Y = torch.Tensor(Y)
    def __getitem__(self, index):


        return self.X[index], self.X[index], self.X[index], self.X_f[index],self.X[index],self.X[index],self.Y[index],self.Y[index],self.Y[index]
    def __len__(self):
        return self.X.size(0)


def frequency_modulate(time, signal, frequency_base, modulation_index):
    # frequency_base 是信号的基础频率
    # modulation_index 是调频指数，它决定了频率的变化程度

    # 生成一个频率随信号变化的频率信号
    modulated_frequency = frequency_base + modulation_index * signal

    # 产生FM调制信号
    modulated_signal = np.cos(2 * np.pi * modulated_frequency * time)
    return modulated_signal

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
    # 原方法 只是降采样
    # signal = pywt.wavedec(signal, 'db4', level=2)[0]


    for i in range(len(signal)):
        _, signal[i] = ecg_to_freq_domain(signal[i], method='fft_amp')
        # _,signal[i] = ecg_to_freq_domain(signal[i],method='fft_complex')
        # _, signal[i] = ecg_to_freq_domain(signal[i], method='psd', nperseg = 512)

    signal = signal.reshape(-1, 1,signal.shape[-1])
    return signal


def ecg_to_freq_domain(signal, fs=360, method='fft_amp', nperseg=None):
    N = len(signal)
    if method == 'fft_amp':
        # FFT幅度谱 (保持原始长度)
        fft_complex = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, 1 / fs)
        spectrum = np.abs(fft_complex) / N * 2
        spectrum[0] /= 2  # 修正直流分量

    elif method == 'fft_complex':
        # 原始复数FFT (科学计算用)
        freqs = np.fft.fftfreq(N, 1 / fs)
        spectrum = np.fft.fft(signal)

    elif method == 'psd':
        # Welch功率谱密度 (通过插值保持长度)
        if nperseg is None:
            nperseg = min(N, 1024)  # 自适应分段
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg, window='hann')
        # 线性插值到原始信号长度
        freqs_interp = np.linspace(0, fs / 2, N)
        spectrum = np.interp(freqs_interp, freqs, psd)
        freqs = freqs_interp
    else:
        raise ValueError("Method must be 'fft_amp', 'fft_complex' or 'psd'")

    return freqs, spectrum
def wavelet_transform_ecg(ecg_signal, wavelet='db6', level=6):
    # 进行小波变换
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # 仅保留细节系数（高频信息），去除逼近系数（低频）
    coeffs[0] = np.zeros_like(coeffs[0])  # 将最低频率部分设为0

    # 进行小波重构，得到频域信号
    freq_signal = pywt.waverec(coeffs, wavelet)

    # 确保输出长度与输入一致
    freq_signal = freq_signal[:len(ecg_signal)]

    return freq_signal
class TransformDataset_SelectChannel:
    def __init__(self, x, y, snr, fs, amplitude, p, min_winsize, is_selectwin, line_num):
        x_length =x.shape[-1]
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

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

        x1 = x.reshape(-1, x_length)
        noisy = trans(x1)
        noisy = noisy.reshape(-1, 1, x_length)

        time = np.linspace(0, 10, x_length)  # 时间范围从0到1，共2500个数据点
        # modulated_signal = frequency_modulate(time, x, 2, 0.1)
        # 新生成方法
        modulated_signal = frequency_modulate2(x,x_length, 2, 0.1, 0.5, 0.5, (0.2, 0.7))

        # 小波变换
        coeffs = pywt.wavedec(x, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        x_wavelet = coeffs[0]
        coeffs = pywt.wavedec(noisy, 'db1', level=1)
        n_wavelet = coeffs[0]
        coeffs = pywt.wavedec(modulated_signal, 'db1', level=1)
        v_wavelet = coeffs[0]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_wavelet = torch.Tensor(x_wavelet)
        self.x_noisy = torch.Tensor(noisy)
        self.n_wavelet = torch.Tensor(n_wavelet)
        self.x_vf = torch.Tensor(modulated_signal)
        self.v_wavelet = torch.Tensor(v_wavelet)

        #构造新标签 正常0，噪声1，FM调频2
        y_noisy = np.full(len(y),1)
        y_vf = np.full(len(y),2)
        self.y_noisy = torch.Tensor(y_noisy)
        self.y_vf = torch.Tensor(y_vf)

    def __getitem__(self, index):
        # 1.原数据 2.小波分解 3.标签 4.噪声 5.噪声+小波分解 6.FM调制信号 7.FM调制信号+小波分解 8.噪声标签 9.FM调制信号标签
        return self.x[index], self.x_wavelet[index], self.y[index], self.x_noisy[index], self.n_wavelet[index], self.x_vf[index], self.v_wavelet[index], self.y_noisy[index], self.y_vf[index]

    def __len__(self):
        return self.x.shape[0]


class TransformDataset_SelectChannel_2:
    def __init__(self, x, y, snr, fs, amplitude, p, min_winsize, is_selectwin, line_num):
                    # val_X, val_Y, 20, 0.05, 0.8, 0.3, 100, False, 1
        x_length =x.shape[-1]  # n,1,2500

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
            # 'line': straight # 会导致全0
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
        x_F = to_frequency(x.copy())
        # 伪噪声数据-频域数据
        noisy_F =to_frequency(noisy.copy())
        # 伪VF数据-频域数据
        modulated_signal_F = to_frequency(modulated_signal.copy())

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
    if opt.dataset == 'vfdb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_vfdb(opt)
    elif opt.dataset == 'mitbih':
        train_data, train_label, val_data, val_label, test_data, test_label = load_mitbih(opt)
    elif opt.dataset == 'cudb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cudb(opt)
    print("[INFO] Train: normal={}".format(train_data.shape), )
    print("[INFO] Val: normal={}".format(val_data.shape), )
    print("[INFO] Test: normal={}".format(test_data.shape), )

    if opt.is_all_data:
        train_X = train_data
        train_Y = train_label
        val_X = val_data
        val_Y = val_label
        test_X = test_data
        test_Y = test_label
    else:
        train_X = train_data[:5000, :, :]
        train_Y = train_label[:5000]
        val_X = val_data[:2000, :, :]
        val_Y = val_label[:2000]
        test_X = test_data[:2000, :, :]
        test_Y = test_label[:2000]

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.model == "SVDD_fake": # 初代不加频域
        train_dataset = TransformDataset_SelectChannel(train_X, train_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    else: # 加频域
        train_dataset = TransformDataset_SelectChannel_2(train_X, train_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
        val_dataset = TransformDataset_SelectChannel_2(val_X, val_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
        test_dataset = TransformDataset_SelectChannel_2(test_X, test_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)

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

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length






