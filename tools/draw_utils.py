import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jiangwei import *

colors1 = '#DE9F83'  # 橘色，透明色
colors2 = '#7494BC'  # 蓝色，透明色
blue = '#76ABDC'
yellow = '#FFE697'
red = '#FC7C7C'

# 特征分布可视化
def plot_tsne_sns(latent, true_labels, opt, display=False):
    df = pd.DataFrame()

    # 二分类画图 将 true_labels 转换为字符串 "Class 0" 和 "Class 1"
    true_labels = ['Class 1' if i == 1 else 'Class 0' for i in true_labels]

    latent = latent.cpu()
    # 降维可视化工具
    pos = tsne(latent)
    # pos = pca(latent)
    # pos = umap(latent)
    # pos = isomap(latent)
    # pos = mds(latent)
    df['x'] = pos[:, 0]
    df['y'] = pos[:, 1]
    df['class'] = true_labels

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    # 使用 scatterplot 来控制点的颜色、边框颜色、透明度等
    sns.scatterplot(data=df,  # 数据源
                    x='x',  # 横轴
                    y='y',  # 纵轴
                    hue="class",  # 分类标签来决定颜色
                    palette={'Class 0': colors2, 'Class 1': colors1},  # 自定义点的颜色
                    s=60,  # 点的大小
                    alpha=0.6,  # 透明度
                    # edgecolor='black',  # 设置黑色边框
                    linewidth=0,  # 边框宽度
                    legend=False)  # 关闭图例框

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne/z5'.format(opt.outf)
        print("save:", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.show()
        plt.close()
        print('Plot')


def plot_tsne_sns2(latent, true_labels, opt, display=False):# 三分类画图
    df = pd.DataFrame()

    latent = latent.cpu() # 0:正常 1：室性 2：噪声
    # 降维可视化工具
    pos = tsne(latent)
    # pos = pca(latent)
    # pos = Umap(latent)
    # pos = isomap(latent)
    # pos = mds(latent)
    df['x'] = pos[:, 0]
    df['y'] = pos[:, 1]

    df['class'] = true_labels
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    # 使用 scatterplot 来控制点的颜色、边框颜色、透明度等
    sns.scatterplot(data=df,  # 数据源
                    x='x',  # 横轴
                    y='y',  # 纵轴
                    hue="class",  # 分类标签来决定颜色
                    palette={0: blue, 1: red, 2: yellow},  # 自定义点的颜色
                    s=100,  # 点的大小
                    alpha=0.6,  # 透明度
                    # edgecolor=edge_color,  # 设置黑色边框
                    # linewidth=1.5,  # 边框宽度
                    legend=False)  # 关闭图例框

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()

    if display:
        plt.show()
    else:
        save_dir = '{}/plot_tsne/z5'.format(opt.outf)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.show()
        plt.close()
        print('Plot')


def plot_hist(scores, true_labels, opt, display=False):
    # scores:异常分数
    # true_labels:标签
    plt.figure()
    plt.grid(False)
    # plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'
    # plt.style.use('seaborn-bright')  # 'seaborn-bright'
    scores = np.array(scores)
    true_labels = np.array(true_labels)
    normal = scores[(true_labels == 0)]
    abnormal = scores[(true_labels == 1)]
    noise = scores[(true_labels == 2)]
    # hrange1 = (0.0005, 0.02)
    # hrange2 = (0, 0.02)
    hrange = (min(scores), max(scores))
    # hrange = (0, 0.5)
    # hrange = (min(scores), 0.1)


    plt.hist(abnormal, 50, facecolor=red, alpha=1,
             label="Abnormal", density=False, range=hrange)
    plt.hist(noise, 50, facecolor=yellow, alpha=1,
             label="Noise", density=False, range=hrange)
    plt.hist(normal, 50, facecolor=blue, alpha=1,
             label="Normal", density=False, range=hrange)

    plt.tick_params(labelsize=12)

    plt.rcParams.update({'font.size': 12})
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.ylim(0,100)
    plt.legend(loc="upper right")

    if display:
        plt.show()
    else:

        save_dir = './plot_hist'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(opt.seed)), transparent=False, bbox_inches='tight')
        plt.show()
        plt.close()
        print('Plot')

