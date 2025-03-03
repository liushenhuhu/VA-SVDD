import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import resample
from scipy.fftpack import fft
from ecg_dataset.load_all_data import load_vfdb
from options import Options
def plot_single_image(signal,index=0,rpeaks=[]):
    # 创建图形
    plt.figure(figsize=(10, 3))
    max = np.max(signal)
    # 设置 y 轴范围
    plt.ylim(-max, max)

    # 绘制信号
    plt.plot(signal, markersize=1, label='ECG Signal')

    # 绘制 R 峰
    # plt.plot(rpeaks, signal[rpeaks], 'ro', markersize=2, label='R-Peaks')

    # 设置 x 轴和 y 轴标签
    plt.xlabel('Time')
    plt.ylabel('ECG Signal')

    # 添加图例
    plt.legend()

    # 保存图像
    # plt.savefig(f'./tools/img/ecg_signal_{index}.png')

    # 显示图像
    plt.show()


opt = Options().parse()
opt.noisetest=True
index = 2


train_data,train_label, val_data, val_label, test_data, test_label =load_vfdb(opt);

sample = train_data
sample = np.asarray(sample, dtype=np.float32)
signal = sample.reshape(-1, sample.shape[-1])
signal_f = fft(signal,axis=1)
# signal_wavelet = pywt.wavedec(signal, 'db1', level=1)

plot_single_image(signal[1])
plot_single_image(signal_f[1])

# print("-"*20,"signal","-"*20)
# print(signal)
# print("-"*20,"signalShape","-"*20)
# print(len(signal))
# print("-"*20,"rpeaks","-"*20)
# print(rpeaks)
# print("-"*20,"nni","-"*20)
# print(nni)
#
# kwargs_welch = {'nfft': 2**12}
# kwargs_ar = {'nfft': 2**10}
# kwargs_lomb= {'nfft': 2**8}
#
# results = fd.frequency_domain(nni=nni, kwargs_welch=kwargs_lomb, kwargs_ar=kwargs_ar, kwargs_lomb=kwargs_lomb)
# print("-"*20,"results","-"*20)
# print(results)




