import matplotlib as plt
import numpy as np
import torch.cpu.amp

from ecg_dataset.ECG_dataloader import TransformDataset_SelectChannel_2, TransformDataset_SelectChannel, to_frequency
from ecg_dataset.load_all_data import load_vfdb, load_cudb, load_mitbih

import matplotlib.pyplot as plt
import os


def plot_all(x, noise, modulated_signal, x_f, noise_f, modulated_signal_f,index):
    # 创建保存图像的目录
    save_dir = 'img'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个包含 6 个子图的画布
    fig, axs = plt.subplots(6, 1, figsize=(10, 12))  # 6行1列的子图，设置合适的图像大小

    # 绘制原始心电数据
    axs[0].plot(x.flatten(), color='#2E75B6')
    axs[0].set_title('Original')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')

    # 绘制噪声数据
    axs[1].plot(noise.flatten(), color='#FFD966')
    axs[1].set_title('Noisy')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')

    # 绘制FM调制的心电数据
    axs[2].plot(modulated_signal.flatten(), color='#FB5B5B')
    axs[2].set_title('FM')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amplitude')

    # 绘制原始心电数据的频域表示
    axs[3].plot(x_f.flatten(), color='#2E75B6')
    axs[3].set_title('Original(Frequency Domain)')
    axs[3].set_xlabel('Frequency')
    axs[3].set_ylabel('Magnitude')

    # 绘制噪声数据的频域表示
    axs[4].plot(noise_f.flatten(), color='#FFD966')
    axs[4].set_title('Noisy(Frequency Domain)')
    axs[4].set_xlabel('Frequency')
    axs[4].set_ylabel('Magnitude')

    # 绘制FM调制数据的频域表示
    axs[5].plot(modulated_signal_f.flatten(), color='#FB5B5B')
    axs[5].set_title('FM(Frequency Domain)')
    axs[5].set_xlabel('Frequency')
    axs[5].set_ylabel('Magnitude')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图像到 img 目录中
    save_path = os.path.join(save_dir, 'ecg_plots'+ str(index).zfill(3)+'.png')
    plt.savefig(save_path)
    plt.close()

    print(f'图像已保存至: {save_path}')


def plot_single(x,y='',index=0):
    # 创建保存图像的目录
    save_dir = 'img/testdb/'+opt.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制原始心电数据
    plt.figure(1,(10,2))
    plt.plot(x.flatten(), color='#2E75B6')
    plt.title(str(y))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)

    # 保存图像到 img 目录中
    save_path = os.path.join(save_dir, 'ecg_plots'+ str(index) + '.svg')
    plt.savefig(save_path)

    # plt.show()
    plt.close()

    print(f'图像已保存至: {save_path}')


if __name__ == '__main__':
    class opt:
        noisetest = True
        # noisetest = False
        seed = 1
        dataset = 'vfdb'
    if opt.dataset == 'vfdb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_vfdb(opt)
    elif opt.dataset == 'mitbih':
        train_data, train_label, val_data, val_label, test_data, test_label = load_mitbih(opt)
    elif opt.dataset == 'cudb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cudb(opt)
    train_X = train_data
    train_Y = train_label

    from ecg_dataset.ECG_dataloader import to_frequency

    #画单个
    # for i,x in enumerate(train_X):
    #     plot_single(x,i)

    # 画vfdb测试集
    # for i,x in enumerate(zip(test_data,test_label)):
    #     plot_single(x[0],x[1], i)
    #     plot_single(to_frequency(x[0]),x[1], i+100)

    # 画全部
    train_dataset = TransformDataset_SelectChannel_2(train_X, train_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
    for i,(x,noise,modulated_signal,x_f,noise_f,modulated_signal_f,_,_,_) in enumerate(train_dataset):
        # print(x_f)
        plot_all(x,noise,modulated_signal,x_f,noise_f,modulated_signal_f,i)