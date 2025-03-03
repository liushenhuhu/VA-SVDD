import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


# 重构效果可视化
def plot_rec(input, output, label, name, opt):
    # input:输入
    # output:输出
    # label:标签

    print(name + " start")

    fig, axs = plt.subplots(10, 8, dpi=100, figsize=(30, 30))
    axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    heat_rec = (np.array(output.tolist()) - np.array(input.tolist())) ** 2

    for i in [17]:
        ax_11 = axs[i].inset_axes([0.0, 0.26, 1, 0.74])
        if label.tolist()[i] == 0:
            color = 'blue'
            input_label = "Normality"
        else:
            color = 'red'
            input_label = "Anomaly"

        ax_11.plot(input.tolist()[i][0], color="blue", linestyle='-', linewidth=1, label="Input")
        ax_11.plot(output.tolist()[i][0], color="black", linestyle='--', linewidth=1, label="Ouput")

        ax_11.set_yticks([])
        if i == 17:
            ax_11.legend(loc='upper left', bbox_to_anchor=(-0.0, 1.5), ncol=3, fontsize=10)

        ax_13 = axs[i].inset_axes([0.0, 0.0, 1, 0.10], sharex=ax_11)
        heat_1 = (np.array(output.tolist()[i][0]) - np.array(input.tolist()[i][0])) ** 2
        heat_norm_1 = (heat_1 - np.min(heat_rec)) / (np.max(heat_rec) - np.min(heat_rec))
        heat_norm_1 = np.reshape(heat_norm_1, (1, -1))

        vmax = np.max(heat_norm_1)

        ax_13.imshow(heat_norm_1, cmap="jet", aspect="auto", vmin=0, vmax=vmax)
        ax_13.set_yticks([])

        ax_11.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_13.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax_13.text(-0.03, 0.5, "Rec", transform=ax_13.transAxes, fontsize=10, va='center', ha='right')

    # plt.show()

    plt.savefig(
        './{}/plot_rec/{}/{}/{}_{}_{}.svg'.format(opt.outf, opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
        transparent=False,
        bbox_inches='tight')
    plt.savefig(
        './{}/plot_rec/{}/{}/{}_{}_{}.pdf'.format(opt.outf, opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
        transparent=False,
        bbox_inches='tight')
    plt.close()
    print('Plot')


# 异常分数分布可视化
def plot_hist(scores, true_labels, opt,name, display=False):
    # scores:异常分数
    # true_labels:标签

    plt.figure()
    plt.grid(False)
    plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'
    # plt.style.use('seaborn-bright')  # 'seaborn-bright'

    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    # hrange1 = (0.0005, 0.02)
    # hrange2 = (0, 0.02)
    hrange = (min(scores), max(scores))
    # hrange = (min(scores), 0.1)

    plt.hist(scores[idx_inliers], 50, facecolor=(0, 0.4, 1, 0.5),  # 浅绿色
             label="Normal", density=False, range=hrange, )
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),  # 浅红色
             label="Abnormal", density=False, range=hrange)

    plt.tick_params(labelsize=22)

    plt.rcParams.update({'font.size': 22})
    plt.xlabel('Anomaly Score', fontsize=22)
    plt.ylabel('Count', fontsize=22)
    plt.legend(loc="upper right")

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_hist/{}'.format('output',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')


def plot_tsne_sns(latent, true_labels, opt, latentname, display=False):
    # latent = latent.cpu()
    # true_labels.cpu()

    pos = TSNE(n_components=2).fit_transform(latent)
    df = pd.DataFrame()
    df['x'] = pos[:, 0]
    df['y'] = pos[:, 1]
    # df['z'] = pos[:, 2]
    legends = list(range(10000))
    df['class'] = [legends[i] for i in true_labels]

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    sns.lmplot(data=df,  # Data source
               x='x',  # Horizontal axis
               y='y',  # Vertical axis
               fit_reg=False,  # Don't fix a regression line
               hue="class",  # Set color,
               legend=False,
               scatter_kws={"s": 25, 'alpha': 0.8})  # S marker size

    # ax.set_zlabel('pca-three')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    plt.xticks([])
    plt.yticks([])

    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()
    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne/{}'.format('output',latentname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')


def plot_ecg_sample(signal, opt, i, display=False):
    # Plot
    fig = plt.figure(figsize=(30, 12), dpi=100)
    x = np.arange(0, 1000, 100)
    x_labels = np.arange(0, 10)

    plt.plot(signal, color='green')

    plt.xticks(x, x_labels)
    plt.xlabel('time (s)', fontsize=16)
    plt.ylabel('value (mV)', fontsize=16)
    fig.tight_layout()
    # plt.savefig("Plot_Hist/11k_{}".format(9999) + '.svg', bbox_inches='tight')
    if display:
        plt.show()
    else:
        save_dir = '{}/plot_ecg_sample/{}'.format(opt.outf, opt.img_data)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(i)), transparent=False, bbox_inches='tight')

        plt.close()
        print('Plot')
