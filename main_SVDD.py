import os
import random

from ecg_dataset.ECG_dataloader import get_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
from options import Options
import numpy as np
import datetime

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

#
DATASETS_NAME = {
    'vfdb': 1,
    # 'mitbih': 1,
    # 'cudb': 1
}

SEEDS = [
    # 0, 1, 2
    # 0, 2, 3
    1, 2, 3, 4, 5, 6,
    # 1
]


if __name__ == '__main__':

    # opt.lr =0.00001
    # opt.batchsize = 32
    opt.nz = 64
    opt.isMasked = False
    opt.is_all_data = True
    opt.augmentation = 'mask'  # mask lr hrv no
    opt.NT = device
    opt.model = 'SVDD'

    opt.alpha = 0 # 分类辅助任务LOSS占比
    opt.sigm = 8 # 防止过拟合
    opt.tf_percent =0.9 # loss中时域占比

    from model.SVDD import ModelTrainer

    opt.augmentation = 'fake'  # mask lr hr no fft
    opt.nz = 64
    opt.nz_m = 32

    darasets_result = {}
    for dataset_name in list(DATASETS_NAME.keys()):
        if  dataset_name == 'vfdb':
            opt.lr = 0.01
            opt.batchsize = 32
        elif dataset_name == 'mitbih':
            opt.lr = 0.005
            opt.batchsize = 128
        elif dataset_name == 'cudb':
            opt.lr = 0.0003
            opt.batchsize = 128
        results_dir = './ECG/{}/{}'.format(opt.model, dataset_name)

        opt.outf = results_dir

        if not os.path.exists(results_dir):
            print("创建:",results_dir)
            os.makedirs(results_dir)
        chk_dir = '{}/{}'.format(results_dir, "model")
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')
        file2print_detail1 = open('{}/results_{}_detail1.log'.format(results_dir, opt.model), 'a+')

        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=file2print)
        print(datetime.datetime.now(), file=file2print_detail)
        print(datetime.datetime.now(), file=file2print_detail1)

        file2print.flush()
        file2print_detail.flush()
        file2print_detail1.flush()

        AUCs = {}
        APs = {}
        Pres = {}
        Recalls = {}
        F1s = {}
        MAX_EPOCHs = {}



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
            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            Pres_seed = {}
            Recalls_seed = {}
            F1s_seed = {}
            model_result = {}

            for seed in SEEDS:

                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                opt.seed = seed
                opt.normal_idx = normal_idx

                dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

                opt.name = "%s/%s" % (opt.model, opt.dataset)
                expr_dir = os.path.join(opt.outf, opt.name, 'train')
                test_dir = os.path.join(opt.outf, opt.name, 'vf-fake-promax')
                if not os.path.isdir(expr_dir):
                    os.makedirs(expr_dir)
                if not os.path.isdir(test_dir):
                    os.makedirs(test_dir)
                args = vars(opt)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')

                print(opt)

                print("################", dataset_name, "##################")
                print("################  Train  ##################")

                model = ModelTrainer(opt, dataloader, device)
                model.c = model.center_c(dataloader["train"])
                ap_test, auc_test, epoch_max_point, pre_test, recall_test, f1_test = model.train()

                print("SEED:{}\t{}\tauc:{:.4f}\tap:{:.4f}\tpre:{:.4f}\trecal:{:.4f}\tf1:{:.4f}\tepoch_max_point:{}".format(seed, dataset_name, auc_test,
                                                                                       ap_test, pre_test, recall_test,
                                                                                       f1_test, epoch_max_point),
                      file=file2print_detail1)
                file2print_detail1.flush()

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                Pres_seed[seed] = pre_test
                Recalls_seed[seed] = recall_test
                F1s_seed[seed] = f1_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

                if opt.model == "ae_hrv":
                    seed_index = "SEED" + str(seed)
                    model_result[seed_index] = {}
                    model_result[seed_index]['encoder'] = model.encoder.state_dict()
                    model_result[seed_index]['decoder'] = model.decoder.state_dict()
                else:
                    seed_index = "SEED" + str(seed)
                    model_result[seed_index] = {}
                    model_result['Backbone'] = model.Backbone.state_dict()
                    model_result['c'] = model.c1
                # torch.save(model_result, chk_dir + '/alpha={}seed={}.pth'.format(opt.alpha, seed))
                # print("保存模型:", chk_dir + '/alpha={}seed={}.pth'.format(opt.alpha, seed))




            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)
            Pres_seed_mean = round(np.mean(list(Pres_seed.values())), 4)
            Pres_seed_std = round(np.std(list(Pres_seed.values())), 4)
            Recalls_seed_mean = round(np.mean(list(Recalls_seed.values())), 4)
            Recalls_seed_std = round(np.std(list(Recalls_seed.values())), 4)
            F1s_seed_mean = round(np.mean(list(F1s_seed.values())), 4)
            F1s_seed_std = round(np.std(list(F1s_seed.values())), 4)

            print(
                "Dataset: {} \t Normal Label: {} \t {} \t AUCs={}+{} \t APs={}+{} \t Pres={}+{} \t Recalls={}+{} \t F1s={}+{}"
                "\t MAX_EPOCHs={}".format(
                    dataset_name, normal_idx, opt.augmentation,
                    AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                    Pres_seed_mean, Pres_seed_std, Recalls_seed_mean, Recalls_seed_std, F1s_seed_mean, F1s_seed_std,
                    MAX_EPOCHs_seed))

            print("{}\t{}\t{}\t{}"
                  "\tAUCs={}+{}\tAPs={}+{}\tPres={}+{}\tRecalls={}+{}\tF1s={}+{}"
                  "\t{}".format(
                opt.model, dataset_name, opt.augmentation, normal_idx,
                AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                Pres_seed_mean, Pres_seed_std, Recalls_seed_mean, Recalls_seed_std, F1s_seed_mean, F1s_seed_std,
                MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            Pres[normal_idx] = Pres_seed_mean
            Recalls[normal_idx] = Recalls_seed_mean
            F1s[normal_idx] = F1s_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\t{}\tTest"
              "\tAUCs={}+{}\tAPs={}+{}\tPres={}+{}\tRecalls={}+{}\tF1s={}+{}"
              "\t{}".format(
            opt.model, dataset_name, opt.augmentation,
            np.mean(list(AUCs.values())), np.std(list(AUCs.values())), np.mean(list(APs.values())),
            np.std(list(APs.values())),
            np.mean(list(Pres.values())), np.std(list(Pres.values())), np.mean(list(Recalls.values())),
            np.std(list(Recalls.values())),
            np.mean(list(F1s.values())), np.std(list(F1s.values())),
            np.max(list(MAX_EPOCHs.values()))
        ), file=file2print)
        darasets_result[dataset_name] = "{}\t{}\t{}\tTest\tAUCs={}+{}\tAPs={}+{}\tPres={}+{}\tRecalls={}+{}\tF1s={}+{}\t{}".format(
            opt.model, dataset_name, opt.augmentation,
            np.mean(list(AUCs.values())), np.std(list(AUCs.values())), np.mean(list(APs.values())),
            np.std(list(APs.values())),
            np.mean(list(Pres.values())), np.std(list(Pres.values())), np.mean(list(Recalls.values())),
            np.std(list(Recalls.values())),
            np.mean(list(F1s.values())), np.std(list(F1s.values())),
            np.max(list(MAX_EPOCHs.values()))
        )
        file2print.flush()
    print("-"*20,"全部结果","-"*20)
    for item in darasets_result.values():
        print(item)