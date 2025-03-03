
import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 16})

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil
from baseline.dataloader.UCR_dataloader import EM_FK

import math

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_plot_sample(samples, idx, identifier, n_samples=6, num_epochs=None,impath=None ,ncol=2):

    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[2]

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    if identifier=="ecg":
        for m in range(nrow):
            for n in range(ncol):
                sample = samples[n * nrow + m, 0, :]
                axarr[m, n].plot(x_points, sample, color=col)
                axarr[m, n].set_ylim(-1, 1)

    else:
        raise Exception("data type error:{}".format(identifier))

    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)

    assert impath is not  None
    fig.savefig(impath)
    plt.clf()
    plt.close()
    return


def save_plot_pair_sample(samples1,samples2, idx, identifier, n_samples=6, num_epochs=None,impath=None ,ncol=2):

    assert n_samples <= samples1.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples1.shape[2] # N,C,L

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    if identifier=="ecg":
        for m in range(nrow):
            sample1=samples1[m,0,:]
            sample2=samples2[m,0,:]

            axarr[m,0].plot(x_points,sample1,color=col)
            axarr[m, 1].plot(x_points, sample2, color=col)
            axarr[m, 0].set_ylim(-1, 1)
            axarr[m, 1].set_ylim(-1, 1)

    else:
        raise Exception("data type error:{}".format(identifier))

    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)

    assert  impath is not None

    fig.savefig(impath)
    plt.clf()
    plt.close()
    return


def plot_tsne(X,y,dim=2):
    tsne = TSNE(n_components=dim, verbose=1, perplexity=40, n_iter=1000)
    x_proj = tsne.fit_transform(X)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f=plt.figure()
    if dim==2:
        ax = f.add_subplot(111)
        ax.scatter(x_proj[:, 0], x_proj[:, 1], lw=0, s=40,c=palette[y.astype(np.int)])
        ax.grid(True)
        for axi in (ax.xaxis, ax.yaxis):
            for tic in axi.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False

    elif dim==3:
        ax = Axes3D(f)
        ax.grid(True)
        ax.scatter(x_proj[:, 0], x_proj[:, 1],x_proj[:,2] ,lw=0, s=40,c=palette[y.astype(np.int)])
        for axi in (ax.xaxis, ax.yaxis,ax.zaxis):
            for tic in axi.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
    f.savefig("sne.png")




def plot_dist(X1,X2,label1,label2,save_dir):
    assert  save_dir is not None
    f=plt.figure()
    ax=f.add_subplot(111)

    # bins = np.linspace(0, 1, 50)
    # _,bins=ax.hist(X1,bins=50)
    # print(bins)
    #
    # if logscale:
    #     bins = np.logspace(np.log10(bins[0]), np.log10(bins[1]), len(bins))

    _, bins, _ = ax.hist(X1, bins=50,range=[0,1],density=True,alpha=0.3,color='r', label=label1)
    _ = ax.hist(X2, bins=bins, alpha=0.3,density=True,color='b',label=label2)
    # ax.set_yticks([])
    ax.legend()
    f.savefig(os.path.join(save_dir, "dist"+label1+label2+".png"))

    #log scale figure
    f_log=plt.figure()
    ax_log=f_log.add_subplot(111)

    log_bins=np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
    _=ax_log.hist(X1, bins=log_bins, range=[0,1],alpha=0.3,density=True,color='r',label=label1)
    _ = ax_log.hist(X2, bins=log_bins,density=True, alpha=0.3,  color='b', label=label2)
    # ax_log.set_yticks([])

    ax_log.legend()
    ax_log.set_xscale('log')
    ax_log.set_xticks([round(x,2) for x in log_bins[::5]])
    ax_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_log.set_xticklabels([round(x,2) for x in log_bins[::5]], rotation=45)
    f_log.savefig(os.path.join(save_dir,"logdist"+label1+label2+".png"))




def save_pair_fig(input,output,save_path):
    '''
    save pair signal (current for first channel)
    :param input: input signal NxL
    :param output: output signal
    :param save_path:
    :return:
    '''
    save_ts_heatmap(input,output,save_path)

    # x_points = np.arange(input.shape[1])
    # fig, ax = plt.subplots(1, 2,figsize=(6, 6))
    # sig_in = input[ 0, :]
    # sig_out=output[0,:]
    # ax[0].plot(x_points, sig_in)
    # ax[1].plot(x_points,sig_out)
    # fig.savefig(save_path)
    # plt.clf()
    # plt.close()


def save_ts_heatmap(input,output,save_path):
    x_points = np.arange(input.shape[1])
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,1]})
    sig_in = input[0, :]
    sig_out = output[0, :]
    ax[0].plot(x_points, sig_in,'k-',linewidth=2.5,label="input signal")
    ax[0].plot(x_points,sig_out,'k--',linewidth=2.5,label="output signal")
    ax[0].set_yticks([])

    ax[0].legend(loc="upper right")



    heat=(sig_out-sig_in)**2
    heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))
    heat_norm=np.reshape(heat_norm,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()
    # fig.show()
    # return
    fig.savefig(save_path)
    plt.clf()
    plt.close()



def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))


    y1 = hist['D_loss']
    y2 = hist['G_loss']

    fig = plt.figure()

    ax1=fig.add_subplot(111)
    ax1.plot(x, y1,'r',label="D_loss")
    ax1.set_ylabel('D_loss')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'b',label="G_loss")

    ax2.set_ylabel('G_loss')

    ax2.set_xlabel('Iter')


    fig.legend(loc='upper left')

    ax1.grid(False)
    ax2.grid(False)

    # fig.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    fig.savefig(path)
    # fig.show()

def save_ts_heatmap_2D(input,output,save_path):
    x_points = np.arange(input.shape[-1])
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,1]})
    sig_in_c1 = np.squeeze(input[0, 0:1,:])
    sig_in_c2 = np.squeeze(input[0, 1:2, :])
    sig_out_c1 = np.squeeze(output[0, 0:1,:])
    sig_out_c2 = np.squeeze(output[0, 1:2, :])
    ax[0].plot(x_points, sig_in_c1,'k-',linewidth=2.5,color= 'blue',label="input signal c1")
    ax[0].plot(x_points, sig_in_c2, 'k-', linewidth=2.5, label="input signal c2")
    ax[0].plot(x_points, sig_out_c1, 'k--', linewidth=2.5,color= 'blue', label="output signal c1")
    ax[0].plot(x_points,sig_out_c2,'k--',linewidth=2.5,label="output signal c2")
    ax[0].set_yticks([])
    plt.tick_params(labelsize=5)

    ax[0].legend(loc="upper right")



    heat=(sig_out_c1-sig_in_c1)**2+(sig_out_c2-sig_in_c2)**2
    heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))
    heat_norm=np.reshape(heat_norm,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()
    # fig.show()
    # return
    fig.savefig(save_path)
    plt.clf()
    plt.close()

#import ecg_plot

def save_ts_heatmap_1D_Batch(input,output,label, save_path):  #(512,1,200)



    #j = 0

    sig_in_batch = []
    sig_out_batch = []

    ecg = []

    #for j in range(64):

    for i in range(0,16):

        sig_in = np.squeeze(input[i, :])
        sig_out = np.squeeze(output[i, :])

        sig_in_batch.append(sig_in)
        sig_out_batch.append(sig_out)

    sig_in_batch = np.squeeze(np.array(sig_in_batch).reshape(1600,-1))
    sig_out_batch = np.squeeze(np.array(sig_out_batch).reshape(1600,-1))



    # np.save('./Plot_Test/input.npy', sig_in_batch)  # (1024, 200)
    # np.save('./Plot_Test/output.npy',sig_out_batch)
    # np.save('./Plot_Test/label.npy', label ) #(1024)

    min_in = np.min(sig_in_batch)
    max_in = np.max(sig_in_batch)
    min_out = np.min(sig_out_batch)
    max_out = np.max(sig_out_batch)
    #
    minn = min(min_in, min_out, max_in, max_out)
    maxx = max(min_in, min_out, max_in, max_out)

    x_points = np.arange(sig_out_batch.shape[-1])
    #fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [6, 1]})
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(24, 6))


    label = label[0]


    # sig_out = sig_out+(np.mean(sig_in)-np.mean(sig_out))




    columns =1
    sample_rate = 100
    secs = len(sig_in_batch)
    #rows = ceil(leads / columns)
    row_height = 1

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (1 / 2) * row_height
    y_max = row_height / 4
    display_factor = 1

    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)

    show_grid = False

    if (show_grid):

        ax[0].set_xticks(np.arange(x_min, x_max, 100))
        ax[0].set_yticks(np.arange(minn, maxx, 0.5))
        #
        ax[0].minorticks_on()

        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))

        ax[0].grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax[0].grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)



    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    ax[0].plot(x_points, sig_in_batch, 'k--', linewidth=2.5, label="input signal")
    ax[0].plot(x_points, sig_out_batch, 'k--', linewidth=2.5, color='blue', label="output signal")



    # ax[0].set_yticks([])
    # ax[0].set_xticks([])

    ax[0].legend(loc="upper right")

    # ecg.append(sig_in_batch[:800])
    # ecg.append(sig_out_batch[:800])
    #
    #   # load data should be implemented by yourself
    # ecg_plot.plot(sig_in_batch[:100], sample_rate=100, title='ECG')
    # ecg_plot.save_as_svg('./Plot_Test/{}_{}'.format(save_path, 0))

    heat = np.abs(sig_out_batch - sig_in_batch)

    # heat = np.mean(heat,dim=1)

    heat_norm = (heat - np.min(heat)) / (np.max(heat) - np.min(heat))


    # for i in range(heat_norm.shape[0]):
    #
    #     if heat_norm[i] > 0.7:
    #
    #         heat_norm[i] = 1
    #
    #     else:
    #
    #         heat_norm[i] = 0

    heat_norm = np.reshape(heat_norm, (1, -1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()

    # plt.xlabel('Channel:{}_iter:{}   Pre:{}'.format(0,epoch, "%.2f%%" % (pre * 100)),font1)  # X轴标签
    if int(label) == 0:
        plt.xlabel('Channel:{}    Label:{}'.format(0, 'Normal'), font1)  # X轴标签
    else:
        plt.xlabel('Channel:{}    Label:{}'.format(0, 'Abnormal'), font1)  # X轴标签

    # fig.show()
    # return
    fig.savefig('./Plot_Test/{}_{}.svg'.format(save_path, 0))
    plt.clf()
    plt.close()


import math

def save_ts_heatmap_1D(input, output,label, save_path, heat_normal):

    j = 0

    initial_state_covariance = 1
    observation_covariance = 1
    initial_value_guess = 0
    transition_matrix = 1
    transition_covariance = 0.9

    x_points = np.arange(input.shape[-1])
    #fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[8,2]})
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,2]})

    sig_filter, _ = EM_FK(initial_value_guess, initial_state_covariance, observation_covariance, transition_covariance, transition_matrix).kalman_1D(input[j, :])
    sig_in = np.squeeze(input[j, :])
    sig_out = np.squeeze(output[j, :])
    sig_filter = np.squeeze(sig_filter)


    label = label[j]

    #sig_out_add = sig_out+(np.round((sig_in[0]-sig_out[0]),1))
    #sig_out = sig_out+(np.mean(sig_in)-np.mean(sig_out))



    # 设置图例并且设置图例的字体及大小
    font1 = {'weight': 'normal',
             'size': 16,}

    ax[0].plot(x_points, sig_in,'k--',linewidth=2.5,label="input signal")
    ax[0].plot(x_points, sig_filter, 'k--', linewidth=2.5, color='orange',label="input signal")
    ax[0].plot(x_points, sig_out,'k--',linewidth=2.5,color='blue',label="output signal")
    print(sig_in)
    print(sig_filter)
    print(sig_out)
    #ax[0].tick_params(labelsize=23)

    ax[0].set_yticks([])
    ax[0].set_xticks([])

    #zax[0].legend(loc="upper right", prop={'size': 23})


    #ax[0].patch.set_facecolor('white')


    heat = np.abs(sig_out - sig_in)   # 输出- 输入


    #heat = np.mean(heat,dim=1)

    for i in range(heat.shape[0]):

        if heat[i] < heat_normal:

            heat[i] = 0


    #heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))

    # for i in range(heat_norm.shape[0]):
    #
    #     if heat_norm[i] > 0.5:
    #
    #         heat_norm[i] = 1


    heat_norm=np.reshape(heat,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect=8)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    fig.tight_layout()

    #plt.xlabel('Channel:{}_iter:{}   Pre:{}'.format(0,epoch, "%.2f%%" % (pre * 100)),font1)  # X轴标签
    if int(label) == 0:
        plt.xlabel('Channel:{}    Label:{}'.format(0,'Normal'),font1)  # X轴标签
    else:
        plt.xlabel('Channel:{}    Label:{}'.format(0,'Abnormal'),font1)  # X轴标签


    #fig.show()
    # return
    fig.savefig('./Plot_4/{}.svg'.format(save_path))
    plt.clf()
    plt.close()
    plt.show()



def save_ts_heatmap_1D_anno(input,output,time_step, anno_step, save_path):


    x_points = np.arange(input.shape[-1])
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,1]})
    sig_in = input[0, :]
    sig_out = output[0, :]
    time_step_in = time_step[0,:]
    anno_step_in = anno_step[0,:]

    time_A = []

    for i in range(anno_step_in.shape[0]):

        if anno_step_in[i] != 0:

           time_A.append(time_step_in[i])


    if (len(time_A) == 1 and time_A[0] < input.shape[-1]) :
        time_A.append(input.shape[-1])

    time_A = np.array(time_A)

    ax[0].plot(x_points, np.squeeze(sig_in),'k-',linewidth=2.5,label="input signal")
    ax[0].plot(x_points,np.squeeze(sig_out),'k--',linewidth=2.5,label="output signal")

    if time_A.shape[0] == 2 and time_A[0]<time_A[1]:

        ax[0].plot(x_points[int(time_A[0]):int(time_A[1])], np.squeeze(sig_in)[int(time_A[0]):int(time_A[1])],'k-',linewidth=2.5,color='red',label="anomaly signal")
        print(time_A)

    ax[0].set_yticks([])

    ax[0].legend(loc="upper right")



    heat=(sig_out-sig_in)**2
    heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))
    heat_norm=np.reshape(heat_norm,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()
    # fig.show()
    # return
    fig.savefig('./Plot/{}_{}.svg'.format(save_path,0))
    plt.clf()
    plt.close()






if __name__ == '__main__':
    import numpy as np
    foo = np.random.normal(loc=1, size=100)  # a normal distribution
    bar = np.random.normal(loc=-1, size=10000)  # a normal distribution
    max_val=max(np.max(foo),np.max(bar))
    min_val=min(np.min(foo),np.min(bar))
    foo=(foo-min_val)/(max_val-min_val)
    bar=(bar-min_val)/(max_val-min_val)
    plot_dist(foo,bar,"1","-1")


