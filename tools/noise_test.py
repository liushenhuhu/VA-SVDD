# 计算噪声数据误判率

import os.path
import random
import statistics
from datetime import datetime

import numpy as np
import torch

from ecg_dataset.ECG_dataloader import get_dataloader
from model.metric_my import evaluate
from options import Options
from tools.draw_utils import plot_tsne_sns, plot_tsne_sns2, plot_hist

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /home/yangliu/project/vf-fake-promax
device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

DATASETS_NAME = {
    'vfdb': 1,
    # 'mitbih':1,
    # 'cudb': 1
}
SEEDS = [
    # 0, 2, 3
    1, 2, 3, 4, 5, 6
    # 1
]

if __name__ == '__main__':
    opt.noisetest = True

    # opt.lr = 0.00001
    # opt.batchsize = 32
    opt.nz = 64
    opt.isMasked = False
    opt.is_all_data = True
    opt.augmentation = 'mask'  # mask lr hrv no
    opt.NT = device
    opt.model = 'SVDD_fake_TF'
    # opt.model = 'SVDD'
    opt.alpha = 0.1  # 分类辅助任务LOSS占比
    opt.sigm = 8  # 防止过拟合
    opt.tf_percent = 0.8  # loss中时域占比

    if opt.model == 'SVDD_fake_TF':
        from model.SVDD_fake_TF import ModelTrainer

        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'SVDD_fake':
        from model.SVDD_fake import ModelTrainer

        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'SVDD':
        from model.SVDD import ModelTrainer

        opt.alpha = 0
        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'SVDD_fake_TF_nonTF':
        from model.SVDD_fake_TF_nonTF import ModelTrainer

        opt.alpha = 0
        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'SVDD_fake_TF_nonClassify':
        from model.SVDD_fake_TF_nonClassify import ModelTrainer

        opt.alpha = 0
        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    else:
        raise Exception("no this model:{}".format(opt.model))

    for dataset_name in list(DATASETS_NAME.keys()):
        if  dataset_name == 'vfdb':
            opt.lr = 0.002
            opt.batchsize = 32
        elif dataset_name == 'mitbih':
            opt.lr = 0.005
            opt.batchsize = 128
        elif dataset_name == 'cudb':
            opt.lr = 0.0003
            opt.batchsize = 128
        print("#" * 10, dataset_name, "#" * 10)
        opt.outf = root_path + "/ECG/{}/{}".format(opt.model, dataset_name)
        for normal_idx in range(DATASETS_NAME[dataset_name]):
            opt.numclass = DATASETS_NAME[dataset_name]
            opt.dataset = dataset_name
            if opt.dataset in ['vfdb']:
                opt.nc = 1
            elif opt.dataset in ['mitbih']:
                opt.nc = 1
            elif opt.dataset in ['cudb']:
                opt.nc = 1

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))
            noise_test_file = open('{}/noise_results_{}.log'.format(root_path, opt.dataset), 'a+')
            print(datetime.now(), file=noise_test_file)
            noisepre_allseeds = []
            pre_allseeds = []
            for seed in SEEDS:
                print("-" * 10, "SEED{}".format(seed), "-" * 10)
                # Set seed
                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                opt.seed = seed
                opt.normal_idx = normal_idx
                dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

                # print(opt)

                seed_index = "SEED" + str(seed)

                model = ModelTrainer(opt, dataloader, device)
                print("loadmodel:",root_path + "/ECG/{}/{}/model/alpha={}seed={}.pth".format(opt.model, opt.dataset,
                                                                                          opt.alpha, seed))
                m = torch.load(
                    root_path + "/ECG/{}/{}/model/alpha={}seed={}.pth".format(opt.model, opt.dataset,
                                                                                          opt.alpha, seed),weights_only=False)
                # m = torch.load(root_path + "/ecg_dataset/ECG/{}/{}/model/alpha={}.pth".format(opt.model,opt.dataset,opt.alpha))
                model.Backbone.load_state_dict(m["Backbone"])
                model.c = m["c"]
                print("save_time",m["save_time"])
                print("save_epoch",m["save_epoch"])
                test_dataloader = model.dataloader["test"]

                with torch.no_grad():
                    all_y_true = []
                    all_y_pred = []


                    latent = torch.empty(0)
                    latent = latent.to(device)
                    for i, data in enumerate(test_dataloader, 0):
                        model.set_input(data, istrain=False)
                        feature1, feature2, feature3, feature4, feature5, feature6, _, _, _,_,_,_ = model.Backbone(
                            model.x,
                            model.x_noisy,
                            model.x_fm,
                            model.x_f,
                            model.x_noisy_f,
                            model.x_fm_f)
                        _, score = model.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6,
                                                        model.c)

                        latent = torch.cat((latent,(feature1+feature4)/2))
                        all_y_true.extend(model.label.cpu().numpy())
                        all_y_pred.extend(score.cpu().numpy())

                    # 画二分类散点图 1
                    # plot_tsne_sns(latent, all_y_true, opt)

                    # 画三分类散点图 2
                    # plot_tsne_sns2(latent, all_y_true, opt)

                    # 画score柱状图
                    # plot_hist(all_y_pred, all_y_true, opt)

                    all_y_true = np.array(all_y_true)
                    all_y_pred = np.array(all_y_pred)

                    Pre, Pre_noise = evaluate(all_y_true, all_y_pred, noisetest=opt.noisetest)

                    # auc_prc, roc_auc, Pre, Recall, f1, _ = evaluate(all_y_true, all_y_pred)
                    # print("auc_prc:{},roc_auc:{},Pre:{},Recall:{},f1:{}".format(auc_prc, roc_auc, Pre, Recall, f1))

                    pre_allseeds.append(Pre)

                    noisepre_allseeds.append(Pre_noise)
                    print("预测正确率：{}".format(Pre))
                    print("噪声预测准确率:{}".format( Pre_noise))
                    print("噪声预测准确率:{}".format( Pre_noise), file=noise_test_file)
        print("#" * 20)
        print("6个随机种子平均准确率为", statistics.mean(pre_allseeds))
        print("6个随机种子平均噪声准确率为", statistics.mean(noisepre_allseeds))
        noise_test_file.flush()
        # 画图
        # y_pred = (all_y_pred >= per)
        # get_plots_1_channel_for_test(test_data, test_label, "vfdb", y_pred)
