import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
# from datasets.transformer import rescale, paa, r_plot, FFT
import matplotlib.pyplot as plt
import pywt
import pickle
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tools.Noisy_Generate import gen_am_noise, gen_bw_noise, gen_fm_noise
import random

LEAD_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_plots_1_channel(fake_ecg, classes, index=0, color='green'):
    fake_ecg_8_chs = fake_ecg.reshape(1, 1000)

    # fake_ecg_8_chs = ecg_filter(fake_ecg_8_chs, fs=100)
    # fake_ecg_8_chs = fake_ecg_8_chs.reshape(1, 2000)
    try:
        # Plot

        fig = plt.figure(figsize=(16, 4), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1]
        for i in range(len(fake_ecg_8_chs)):
            plt.subplot(1, 1, idx[i])
            plt.plot(fake_ecg_8_chs[i], color=color, label=str(i))
            # plt.title(LEAD_8[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=38)
            plt.ylabel('value (mV)', fontsize=38)

        fig.tight_layout()
        plt.tick_params(labelsize=38)
        plt.savefig('./Plot_Noisy/{}_{}'.format(classes, index) + '.svg', bbox_inches='tight', dpi=200)
        # plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def get_plots_12_channel(fake_ecg, color='green'):
    # fake_ecg_8_chs = fake_ecg.reshape(8, 2000)

    # ake_ecg_8_chs = ecg_filter(fake_ecg_8_chs, fs=100)
    try:
        # Plot

        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        for i in range(len(fake_ecg)):
            plt.subplot(6, 2, idx[i])
            plt.plot(fake_ecg[i], color=color, label=str(i))
            plt.title(LEAD_12[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        # plt.savefig('./Plot/plot_CinC2011_epoch{}'.format(epoch) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def Am_Noisy(x, fs, amplitude, min_winsize=None, is_selectwin=None):  # snr:信噪比

    # print("Am")
    Am_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 5
    for i in range(x.shape[0]):
        Am_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)
        x_t_win = np.arange(win_start_point, win_end_point, 1) / 5

        for j in range(x.shape[1]):

            if is_selectwin == False:

                signal = np.array(x[i][j])
                _, am_noisy = gen_am_noise(x_t, signal, fs, amplitude)
                Am_single.append(am_noisy)

            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                _, win_am_noisy = gen_am_noise(x_t_win, win_signal, fs, amplitude)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = win_am_noisy
                Am_single.append(signal_add_win_noisy)

        Am_single = np.array(Am_single)
        # get_plots_12_channel(Am_single)
        # get_plots_12_channel(x[i])
        Am_all.append(Am_single)

    Am_all = np.array(Am_all)

    return Am_all


def Am_Noisy_1D(x, fs, amplitude, is_selectwin=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Am")
    Am_all = []
    x_t = np.arange(0, len(x), 1) / 5

    len_signal = len(x)
    # win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)
    x_t_win = np.arange(win_start_point, win_end_point, 1) / 5

    if is_selectwin == False:

        signal = np.array(x)
        _, am_noisy = gen_am_noise(x_t, signal, fs, amplitude)
        Am_single = am_noisy

    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        _, win_am_noisy = gen_am_noise(x_t_win, win_signal, fs, amplitude)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = win_am_noisy
        Am_single = signal_add_win_noisy
    # get_plots_12_channel(Am_single)
    # get_plots_12_channel(x[i])

    return Am_single


def Bw_Noisy(x, fs, amplitude, min_winsize=None, is_selectwin=None):  # snr:信噪比

    # print("Bw")
    Bw_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 5
    len_signal = len(x[0][0])

    for i in range(x.shape[0]):

        Bw_single = []
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)
        x_t_win = np.arange(win_start_point, win_end_point, 1) / 5
        for j in range(x.shape[1]):

            if is_selectwin == False:
                signal = np.array(x[i][j])
                _, bw_noisy = gen_bw_noise(x_t, signal, fs, amplitude)
                Bw_single.append(bw_noisy)
            else:

                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                _, win_bw_noisy = gen_bw_noise(x_t_win, win_signal, fs, amplitude)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = win_bw_noisy
                Bw_single.append(signal_add_win_noisy)

        Bw_single = np.array(Bw_single)
        # get_plots_1_channel(x[i][j])
        # get_plots_1_channel(signal)
        # get_plots_1_channel(Bw_single)
        Bw_all.append(Bw_single)

    Bw_all = np.array(Bw_all)

    return Bw_all


def Bw_Noisy_1D(x, fs, amplitude, is_selectwin=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Bw")
    Bw_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 5
    x_t_win = np.arange(win_start_point, win_end_point, 1) / 5

    if is_selectwin == False:
        signal = np.array(x)
        _, bw_noisy = gen_bw_noise(x_t, signal, fs, amplitude)
        Bw_single = bw_noisy
    else:

        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        _, win_bw_noisy = gen_bw_noise(x_t_win, win_signal, fs, amplitude)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = win_bw_noisy
        Bw_single = signal_add_win_noisy

        # get_plots_1_channel(x[i][j])
        # get_plots_1_channel(signal)
        # get_plots_1_channel(Bw_single)

    return Bw_single


def Fm_Noisy(x, fs, amplitude, min_winsize=None, is_selectwin=None):  # snr:信噪比

    # print("Fm")
    Fm_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 50
    len_signal = len(x[0][0])

    for i in range(x.shape[0]):

        Fm_single = []
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)
        x_t_win = np.arange(win_start_point, win_end_point, 1) / 5
        for j in range(x.shape[1]):

            if is_selectwin == False:
                signal = np.array(x[i][j])
                _, fm_noisy = gen_fm_noise(x_t, signal, fs, amplitude)
                Fm_single.append(fm_noisy)
            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                _, win_fm_noisy = gen_fm_noise(x_t_win, win_signal, fs, amplitude)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = win_fm_noisy
                Fm_single.append(signal_add_win_noisy)

        Fm_single = np.array(Fm_single)
        # get_plots_12_channel(signal[:12])
        # get_plots_12_channel(Fm_single[:12])
        Fm_all.append(Fm_single)

    Fm_all = np.array(Fm_all)

    return Fm_all


def Fm_Noisy_1D(x, fs, amplitude, is_selectwin=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Fm")
    Fm_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 50

    x_t_win = np.arange(win_start_point, win_end_point, 1) / 5

    if is_selectwin == False:
        signal = np.array(x)
        _, fm_noisy = gen_fm_noise(x_t, signal, fs, amplitude)
        Fm_single = fm_noisy
    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        _, win_fm_noisy = gen_fm_noise(x_t_win, win_signal, fs, amplitude)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = win_fm_noisy
        Fm_single = signal_add_win_noisy

    # get_plots_12_channel(signal[:12])
    # get_plots_12_channel(Fm_single[:12])

    return Fm_single


def Bw_Am_Fm_Noisy(x, fs_bw, amplitude_bw, fs_am, amplitude_am, fs_fm, amplitude_fm):  # snr:信噪比

    print("Bw_Am_Fm")
    Fm_all = []
    x_t = np.arange(0, x.shape[-1], 1) / 5
    for i in range(x.shape[0]):

        Fm_single = []
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            bw_t, bw_noisy = gen_bw_noise(x_t, signal, fs_bw, amplitude_bw)
            am_t, am_noisy = gen_am_noise(bw_t, bw_noisy, fs_am, amplitude_am)
            fm_t, fm_noisy = gen_fm_noise(am_t, am_noisy, fs_fm, amplitude_fm)
            Fm_single.append(fm_noisy)

        Fm_single = np.array(Fm_single)
        # get_plots_8_channel(Fm_single, i)
        Fm_all.append(Fm_single)

    Fm_all = np.array(Fm_all)

    return Fm_all


# def Gamma_Noisy(x, snr):   # snr:信噪比
#
#     print('Gamma')
#     x_gamma = []
#     x_gamma_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         gamma = np.random.gamma(shape= 1, size = len(signal)) * np.sqrt(npower)
#
#         x_gamma.append(x[i] + gamma)
#         x_gamma_only.append(gamma)
#
#     x_gamma = np.array(x_gamma)
#     x_gamma_only = np.array(x_gamma_only)
#
#     return x_gamma, x_gamma.shape[-1]


def Gamma_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Gamma")

    Gamma_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        Gamma_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):

            if is_selectwindow == False:
                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                gamma = np.random.gamma(shape=1, size=len(signal)) * np.sqrt(npower)

                Gamma_single.append(x[i][j] + gamma)

            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                gamma = np.random.gamma(shape=1, size=len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][win_start_point:win_end_point] + gamma
                Gamma_single.append(signal_add_win_noisy)

        Gamma_single = np.array(Gamma_single)
        Gamma_all.append(Gamma_single)

    Gamma_all = np.array(Gamma_all)

    return Gamma_all


def Gamma_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Gamma")

    Gamma_all = []
    snr = 10 ** (snr / 10.0)

    if is_selectwindow == False:
        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        gamma = np.random.gamma(shape=1, size=len(signal)) * np.sqrt(npower)

        Gamma_single = (x + gamma)

    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        gamma = np.random.gamma(shape=1, size=len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + gamma
        Gamma_single = (signal_add_win_noisy)

    return Gamma_single


# def Rayleign_Noisy(x, snr):   # snr:信噪比
#
#     print('Ralyeign')
#     x_rayleign = []
#     x_rayleign_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         rayleign = np.random.rayleigh(size = len(signal)) * np.sqrt(npower)
#
#         x_rayleign.append(x[i] + rayleign)
#         x_rayleign_only.append(rayleign)
#
#     x_rayleign = np.array(x_rayleign)
#     x_rayleign_only = np.array(x_rayleign_only)
#
#     return  x_rayleign, x_rayleign.shape[-1]


def Rayleign_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Rayleign")

    Rayleign_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        Rayleign_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):

            if is_selectwindow == False:
                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                rayleign = np.random.rayleigh(size=len(signal)) * np.sqrt(npower)

                Rayleign_single.append(x[i][j] + rayleign)
            else:

                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                rayleign = np.random.rayleigh(size=len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][win_start_point:win_end_point] + rayleign
                Rayleign_single.append(signal_add_win_noisy)

        Rayleign_single = np.array(Rayleign_single)
        Rayleign_all.append(Rayleign_single)
        # get_plots_12_channel(Rayleign_single)
        # get_plots_12_channel(x[i])

    Rayleign_all = np.array(Rayleign_all)

    return Rayleign_all


def Rayleign_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Rayleign")

    Rayleign_all = []
    snr = 10 ** (snr / 10.0)

    Rayleign_single = []
    len_signal = len(x)

    if is_selectwindow == False:
        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        rayleign = np.random.rayleigh(size=len(signal)) * np.sqrt(npower)

        Rayleign_single = (x + rayleign)
    else:

        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        rayleign = np.random.rayleigh(size=len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + rayleign
        Rayleign_single = (signal_add_win_noisy)

    return Rayleign_single


# def Exponential_Noisy(x, snr):   # snr:信噪比
#
#     print("Exponential")
#     x_exponential = []
#     x_exponential_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         exponential = np.random.exponential(size = len(signal)) * np.sqrt(npower)
#
#         x_exponential.append(x[i] + exponential)
#         x_exponential_only.append(exponential)
#
#     x_exponential = np.array(x_exponential)
#     x_exponential_only = np.array(x_exponential_only)
#
#     return x_exponential, x_exponential.shape[-1]


def Exponential_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Exponential")

    exponential_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        exponential_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):
            if is_selectwindow == False:
                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                exponential = np.random.exponential(size=len(signal)) * np.sqrt(npower)

                exponential_single.append(x[i][j] + exponential)
            else:

                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                exponential = np.random.exponential(size=len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][
                                                                      win_start_point:win_end_point] + exponential
                exponential_single.append(signal_add_win_noisy)

        exponential_single = np.array(exponential_single)
        exponential_all.append(exponential_single)

    exponential_all = np.array(exponential_all)

    return exponential_all


def Exponential_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Exponential")

    exponential_all = []
    snr = 10 ** (snr / 10.0)

    exponential_single = []
    len_signal = len(x)

    if is_selectwindow == False:
        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        exponential = np.random.exponential(size=len(signal)) * np.sqrt(npower)

        exponential_single = (x + exponential)
    else:

        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        exponential = np.random.exponential(size=len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + exponential
        exponential_single = (signal_add_win_noisy)

    return exponential_single


# def Uniform_Noisy(x, snr):   # snr:信噪比
#
#     print("Uniform")
#     x_uniform = []
#     x_uniform_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         uniform = np.random.uniform(size = len(signal)) * np.sqrt(npower)
#
#         x_uniform.append(x[i] + uniform)
#         x_uniform_only.append(uniform)
#
#     x_uniform = np.array(x_uniform)
#     x_uniform_only = np.array(x_uniform_only)
#
#     return x_uniform, x_uniform.shape[-1]


def Uniform_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Uniform")

    uniform_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        uniform_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):

            if is_selectwindow == False:

                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                uniform = np.random.uniform(size=len(signal)) * np.sqrt(npower)

                uniform_single.append(x[i][j] + uniform)
            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                uniform = np.random.uniform(size=len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][win_start_point:win_end_point] + uniform
                uniform_single.append(signal_add_win_noisy)

        uniform_single = np.array(uniform_single)
        uniform_all.append(uniform_single)

    uniform_all = np.array(uniform_all)

    return uniform_all


def Uniform_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Uniform")

    uniform_all = []
    snr = 10 ** (snr / 10.0)

    uniform_single = []
    len_signal = len(x)

    if is_selectwindow == False:

        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        uniform = np.random.uniform(size=len(signal)) * np.sqrt(npower)

        uniform_single = (x + uniform)
    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        uniform = np.random.uniform(size=len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + uniform
        uniform_single = (signal_add_win_noisy)

    return uniform_single


# def Poisson_Noisy(x, snr):   # snr:信噪比
#
#     print("possion")
#     x_poisson = []
#     x_poisson_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         poisson = np.random.poisson(2, len(signal)) * np.sqrt(npower)
#
#         x_poisson.append(x[i] + poisson)
#         x_poisson_only.append(poisson)
#
#     x_poisson = np.array(x_poisson)
#     x_poisson_only = np.array(x_poisson_only)
#
#     return x_poisson, x_poisson.shape[-1]


def Poisson_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Poisson")

    poisson_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        poisson_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):

            if is_selectwindow == False:
                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                poisson = np.random.poisson(2, len(signal)) * np.sqrt(npower)

                poisson_single.append(x[i][j] + poisson)

            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                poisson = np.random.poisson(2, len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][win_start_point:win_end_point] + poisson
                poisson_single.append(signal_add_win_noisy)

        poisson_single = np.array(poisson_single)
        poisson_all.append(poisson_single)

    poisson_all = np.array(poisson_all)

    return poisson_all


def Poisson_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Poisson")

    poisson_all = []
    snr = 10 ** (snr / 10.0)

    len_signal = len(x)

    if is_selectwindow == False:
        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        poisson = np.random.poisson(2, len(signal)) * np.sqrt(npower)

        poisson_single = (x + poisson)

    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        poisson = np.random.poisson(2, len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + poisson
        poisson_single = (signal_add_win_noisy)

    return poisson_single


# def Gussian_Noisy(x, snr):   # snr:信噪比
#
#     print("Gussian")
#     x_gussian = []
#     x_gussian_only = []
#     snr = 10 ** (snr / 10.0)
#     for i in range(x.shape[0]):
#
#         signal = np.array(x[i])
#         signal = np.squeeze(signal)
#         xpower = np.sum(signal ** 2) / len(signal)
#         npower = xpower / snr
#         gussian = np.random.randn(len(signal)) * np.sqrt(npower)
#
#         x_gussian.append(x[i] + gussian)
#         x_gussian_only.append(gussian)
#
#     x_gussian = np.array(x_gussian)
#     x_gussian_only = np.array(x_gussian_only)
#
#     return x_gussian, x_gussian.shape[-1]


def Gussian_Noisy(x, snr, min_winsize=None, is_selectwindow=None):  # snr:信噪比

    # print("Gussian")

    gussian_all = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        gussian_single = []
        len_signal = len(x[0][0])
        win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

        for j in range(x.shape[1]):

            if is_selectwindow == False:

                signal = np.array(x[i][j])
                xpower = np.sum(signal ** 2) / len(signal)
                npower = xpower / snr
                gussian = np.random.randn(len(signal)) * np.sqrt(npower)
                gussian_single.append(x[i][j] + gussian)
                # get_plots_1_channel(x[i][j])
                # get_plots_1_channel(x[i][j] + gussian)
            else:
                signal = np.array(x[i][j])
                win_signal = signal[win_start_point: win_end_point]
                xpower = np.sum(win_signal ** 2) / len(win_signal)
                npower = xpower / snr
                gussian = np.random.randn(len(win_signal)) * np.sqrt(npower)
                signal_add_win_noisy = signal
                signal_add_win_noisy[win_start_point:win_end_point] = x[i][j][win_start_point:win_end_point] + gussian
                gussian_single.append(signal_add_win_noisy)
                # get_plots_1_channel(x[i][j])
                # get_plots_1_channel(signal_add_win_noisy)

        gussian_single = np.array(gussian_single)
        gussian_all.append(gussian_single)

    gussian_all = np.array(gussian_all)

    return gussian_all


def Gussian_Noisy_1D(x, snr, is_selectwindow=None, win_start_point=None, win_end_point=None):  # snr:信噪比

    # print("Gussian")

    gussian_all = []
    snr = 10 ** (snr / 10.0)

    # len_signal = len(data[0])
    # win_start_point, win_end_point = Select_Win(min_winsize, len_signal, i)

    if is_selectwindow == False:

        signal = np.array(x)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        gussian_single = (x + gussian)
        # get_plots_1_channel(x[i][j])
        # get_plots_1_channel(x[i][j] + gussian)
    else:
        signal = np.array(x)
        win_signal = signal[win_start_point: win_end_point]
        xpower = np.sum(win_signal ** 2) / len(win_signal)
        npower = xpower / snr
        gussian = np.random.randn(len(win_signal)) * np.sqrt(npower)
        signal_add_win_noisy = signal
        signal_add_win_noisy[win_start_point:win_end_point] = x[win_start_point:win_end_point] + gussian
        gussian_single = (signal_add_win_noisy)
        # get_plots_1_channel(x[i][j])
        # get_plots_1_channel(signal_add_win_noisy)

    return gussian_single


def StraightLine(x, num):
    channel_num = x.shape[1]
    signal = []

    for i in range(x.shape[0]):

        channel_line = random.sample(range(0, channel_num), num)
        single = []
        for j in range(channel_num):

            if j in channel_line:

                single.append(x[i][j] * 0)
            else:
                single.append(x[i][j])
        single = np.array(single)
        signal.append(single)
    signal = np.array(signal)
    return signal


# def StraightLine_1D(x, is_select_win, win_start_point, win_end_point):
#
#     signal = x
#     # if is_select_win == True:
#     #     signal[win_start_point: win_end_point] = x[win_start_point:win_end_point]*0
#     # else:
#     signal = x*0
#
#     return signal


def StraightLine_1D(x, is_select_win, win_start_point, win_end_point):
    signal = x
    if is_select_win == True:
        signal[win_start_point: win_end_point] = x[win_start_point:win_end_point] * 0
    else:
        signal = x * 0

    return signal


def Select_Win(min_win, len, seed):
    random.seed(seed)
    start_index = random.randint(0, len - min_win)

    # end_index = random.randint(start_index+min_win, len)
    end_index = start_index + min_win

    return start_index, end_index
