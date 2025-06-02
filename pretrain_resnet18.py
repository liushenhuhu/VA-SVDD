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
    'mitbih': 1,
    'cudb': 1
}

SEEDS = [
    # 0, 1, 2
    # 0, 2, 3
    1, 2, 3, 4, 5, 6,
    # 1
]

# 主实验
if __name__ == '__main__':

    # opt.lr =0.00001
    # opt.batchsize = 32
    opt.nz = 32
    opt.isMasked = False
    opt.is_all_data = True
    opt.augmentation = 'mask'  # mask lr hrv no
    opt.NT = device
    opt.model = 'resnet18'

    opt.alpha = 0.1 # 分类辅助任务LOSS占比
    opt.sigm = 8 # 防止过拟合
    opt.tf_percent =0.8 # loss中时域占比

    opt.batchsize = 32
    opt.lr = 0.1

    if opt.model == 'resnet18':
        from model.resnet import ModelTrainer
        opt.augmentation = 'fake'  # mask lr hr no fft
        opt.nz = 64
        opt.nz_m = 32
    else:
        raise Exception("no this model:{}".format(opt.model))

    darasets_result = {}
    for dataset_name in list(DATASETS_NAME.keys()):
        results_dir = './ECG/{}/{}'.format(opt.model, dataset_name)

        opt.outf = results_dir
        if not os.path.exists(results_dir):
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
                pre= model.train()

                print("SEED:{}\t{}\tap:{}".format(seed, dataset_name, pre),
                      file=file2print_detail1)
                file2print_detail1.flush()


                Pres_seed[seed] = pre


                seed_index = "SEED" + str(seed)
                model_result['teacher_T'] = model.Backbone.teacher_T.state_dict()
                model_result['teacher_F'] = model.Backbone.teacher_F.state_dict()


                torch.save(model_result, chk_dir + '/seed={}.pth'.format(seed))




            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)

            Pres_seed_mean = round(np.mean(list(Pres_seed.values())), 4)
            print("平均pre：",Pres_seed_mean)
            darasets_result[dataset_name] = Pres_seed_mean
    print("-"*20,"全部结果","-"*20)
    for item in darasets_result.values():
        print(item)