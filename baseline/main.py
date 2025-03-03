import os
import torch
from baseline.options import Options

# from baseline.dataloader.UCR_dataloader import load_data, EM_FK
from ecg_dataset.ECG_dataloader import get_dataloader
import numpy as np
# from baseline.dataloader.UCR_dataloader_goad import load_trans_data_ucr_affine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = Options().parse()
opt.device=device


opt.batchsize = 128
opt.nc = 1
opt.nz = 64
opt.lr = 0.001
opt.Snr = 75
opt.model = 'AE_CNN_4indicator'

if opt.model == "BeatGAN_4indicator":
    from baseline.model.BeatGAN_4indicator import BeatGAN as ModelTrainer
elif opt.model == "Ganomaly_4indicator":
    from baseline.model.Ganomaly_4indicator import Ganomaly as ModelTrainer
elif opt.model == "AE_CNN_4indicator":
    from baseline.model.AE_CNN_4indicator import ModelTrainer
elif opt.model == 'USAD_4indicator':
    from baseline.model.USAD_4indicator import UsadModel,training,testing
elif opt.model=='DeepSVDD':
    from baseline.model.DeepSVDD import DeepSVDD as ModelTrainer
else:
    #raise Exception("no this model_eeg :{}".format(opt.model))
    print(opt.model)



DATASETS_NAME={
   # 'cpsc2021':1,
   #  'ptbxl':1
   #  'ZZU':1
   # 'icentia11k': 1,
    'vfdb':1, #只用vfdb
}
SEEDS=[
    0,2,3,
]
# lead=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
import neurokit2 as nk
if __name__ == '__main__':

    results_dir='./logs_dualchannel_mask'

    opt.outf = results_dir
    opt.MI='AMI'
    print(opt.model)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if opt.model in ['AE_CNN_noisy_multi', 'AE_CNN_Noisy']:

        file2print = open('{}/results_{}_{}-{}.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')
        file2print_detail = open('{}/results_{}_{}-{}_detail.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')

    elif opt.model in ['AE_CNN_Filter']:

        file2print = open('{}/results_{}_{}.log'.format(results_dir, opt.model, opt.FilterType), 'a+')
        file2print_detail = open('{}/results_{}_{}.detail.log'.format(results_dir, opt.model, opt.FilterType), 'a+')


    else:

        file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')

    import datetime
    print(datetime.datetime.now())
    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)

    if opt.model in ['AE_CNN_noisy_multi', 'AE_CNN_Noisy']:

        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr))
        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print_detail)
        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAUC_MAX\tAP_mean\tAP_std\tAP_MAX\tPre_mean\tPre_std\tPre_MAX\tRec_mean\tRec_std\tRec_MAX\tMax_Epoch", file=file2print_detail)

    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAUC_MAX\tAP_mean\tAP_std\tAP_MAX\tPre_mean\tPre_std\tPre_MAX\tRec_mean\tRec_std\tRec_MAX\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tPre_mean\tPre_std\tRec_mean\tRec_std\tMax_Epoch", file=file2print)
    file2print.flush()
    file2print_detail.flush()

    for dataset_name in list(DATASETS_NAME.keys()):
        AUCs={}
        APs={}
        MAX_EPOCHs = {}
        Pres={}
        Recs={}

        for normal_idx in range(DATASETS_NAME[dataset_name]): #只有vfdb
            # normal_idx = 1
            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))
            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            Pre_seed = {}
            Rec_seed = {}
            for seed in SEEDS:
                opt.seed = seed

                opt.normal_idx = normal_idx
                opt.dataset = dataset_name
                # dataloader, opt.isize, opt.signal_length = load_data(opt,dataset_name, None, True)
                dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

                opt.nc = 1

                opt.name = "%s/%s" % (opt.model, opt.dataset)
                expr_dir = os.path.join(opt.outf, opt.name, 'train')
                test_dir = os.path.join(opt.outf, opt.name, 'test')

                if not os.path.isdir(expr_dir):
                    os.makedirs(expr_dir)
                if not os.path.isdir(test_dir):
                    os.makedirs(test_dir)


                print("################", dataset_name, "##################")
                print("################  Train  ##################")

                if opt.model == 'USAD':

                    model = UsadModel(opt).to(device)

                    ap_test, auc_test, epoch_max_point = training(opt.niter, model, dataloader['train'], dataloader['val'], dataloader['test'])

                # elif opt.model == 'GOAD':
                #     x_train, x_val, y_val, x_test, y_test, ratio_val, ratio_test = load_trans_data_ucr_affine(opt,dataset_name)
                #     model = Goad(opt)  # tc_obj = tc.TransClassifierTabular(args)
                #     ap_test, auc_test, epoch_max_point = model.train(x_train, x_val, y_val, x_test, y_test, ratio_val, ratio_test)

                elif opt.model == 'AE_CNN_Filter':

                    if opt.FilterType =='Kalman':

                        model = ModelTrainer(opt, dataloader, device)

                        ap_test, auc_test, epoch_max_point = model.train()

                    # elif opt.FilterType =='EM-KF':
                    #
                    #     model = ModelTrainer(opt, dataloader, device)
                    #     Filter_EM = model.pre_train()
                    #
                    #     dataloader, opt.isize, opt.signal_length = load_data(opt, dataset_name, Filter_EM, False)
                    #
                    #     model = ModelTrainer(opt, dataloader, device)
                    #
                    #     if opt.istest == True:
                    #
                    #         print('Testing')
                    #
                    #         model.G.load_state_dict(torch.load('./Model_CheckPoints2/{}_{}_{}_{}.pkl'.format(opt.model, dataset_name, normal_idx, seed)))
                    #         ap_test, auc_test, _, _ = model.test()
                    #         epoch_max_point = 0
                    #
                    #     else:
                    #
                    #         ap_test, auc_test, epoch_max_point = model.train()

                elif opt.model == 'USAD_4indicator':
                    model = UsadModel(opt).to(device)
                    if opt.istest == True:
                        print('Testing')
                        model=torch.load(
                            './Model_CheckPoint/{}_{}_{}_{}.pkl'.format(opt.model, dataset_name, normal_idx, seed),
                            map_location='cpu')
                        #model.eval()
                        model=model.to(device)
                        ap_test, auc_test, Pre_test, Rec_test, f1 = testing(model, opt, dataloader["test"])
                        epoch_max_point = 0
                    else:
                        ap_test, auc_test,Pre_test,Rec_test, epoch_max_point = training(opt, model, dataloader['train'],
                                                                      dataloader['val'], dataloader['test'])
                elif opt.model=='DeepSVDD':
                    model = ModelTrainer()
                    model.train()
                else:

                    model = ModelTrainer(opt, dataloader, device)
                    #model = ModelTrainer(opt)


                    if opt.istest == True:
                        print('Testing')
                        if opt.model == 'BeatGAN_4indicator':
                            model.G.load_state_dict(torch.load(
                                '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(
                                    opt.model, opt.dataset, opt.normal_idx, opt.seed), map_location='cpu'))
                        elif opt.model == 'Ganomaly_4indicator':
                            model.netg.load_state_dict(torch.load(
                                '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(
                                    opt.model, opt.dataset, opt.normal_idx, opt.seed),
                                map_location='cpu'))
                        elif opt.model == 'AE_CNN_4indicator':
                            model.G.load_state_dict(torch.load(
                                '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(
                                    opt.model, opt.dataset, opt.normal_idx, opt.seed), map_location='cpu'))
                            # model = torch.load(
                            #     '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(
                            #         opt.model, opt.dataset, opt.normal_idx, opt.seed), map_location='cpu')
                        # model.netg.eval()

                        # print(next(model.G.parameters()))
                        # ap_test, auc_test, _, Pre_test, Rec_test, f1 = model.test()
                        ap_test, auc_test, Pre_test, Rec_test, f1 = model.test()
                        # rocprc, rocauc, best_th, Pre, Rec, f1
                        # print(auc_test)
                        epoch_max_point = 0
                        # if opt.isInterpret==True:
                        #     interpret_ptbxl_multi(model.netg,dataloader['test'],label,pred)
                    else:
                        #ap_test, auc_test, epoch_max_point = model.train()
                        ap_test, auc_test, Pre_test, Rec_test, epoch_max_point= model.train()

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                MAX_EPOCHs_seed[seed] = epoch_max_point
                Pre_seed[seed] = Pre_test
                Rec_seed[seed] = Rec_test

                # End For
            try:
                MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
                AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
                AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
                AUCs_seed_MAX = round(np.max(list(AUCs_seed.values())), 4)
                APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
                APs_seed_std = round(np.std(list(APs_seed.values())), 4)
                APs_seed_MAX = round(np.max(list(APs_seed.values())), 4)
                # Pres_seed_mean = round(np.mean(list(Pre_seed.values())), 4)
                # Pres_seed_std = round(np.std(list(Pre_seed.values())), 4)
                # Pres_seed_MAX = round(np.max(list(Pre_seed.values())), 4)
                # Recs_seed_mean = round(np.mean(list(Rec_seed.values())), 4)
                # Recs_seed_std = round(np.std(list(Rec_seed.values())), 4)
                # Recs_seed_MAX = round(np.max(list(Rec_seed.values())), 4)

                # print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t AUC_MAX={} \t APs={}+{} \t AP_MAX={} \t Pres={}+{} \t  Pres_MAX={}  \tRecs={}+{}\t Recs_MAX={} \t  MAX_EPOCHs={}".format(
                #     dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std,AUCs_seed_MAX, APs_seed_mean, APs_seed_std,APs_seed_MAX,Pres_seed_mean,Pres_seed_std,Pres_seed_MAX,Recs_seed_mean,Recs_seed_std,Recs_seed_MAX, MAX_EPOCHs_seed))
                #
                # print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                #     opt.model, dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, AUCs_seed_MAX, APs_seed_mean, APs_seed_std,APs_seed_MAX,Pres_seed_mean,Pres_seed_std,Pres_seed_MAX,Recs_seed_mean,Recs_seed_std,Recs_seed_MAX, MAX_EPOCHs_seed)
                #    , file=file2print_detail)
                # file2print_detail.flush()

                print(
                    "Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t AUC_MAX={} \t APs={}+{} \t AP_MAX={}  \t  MAX_EPOCHs={}".format(
                        dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, AUCs_seed_MAX, APs_seed_mean,
                        APs_seed_std, APs_seed_MAX,  MAX_EPOCHs_seed))

                print(
                    "{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                        opt.model, dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, AUCs_seed_MAX,
                        APs_seed_mean, APs_seed_std, APs_seed_MAX,  MAX_EPOCHs_seed)
                    , file=file2print_detail)
                file2print_detail.flush()

                AUCs[normal_idx] = AUCs_seed_mean
                APs[normal_idx] = APs_seed_mean
                MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

                # Pres[normal_idx] = Pres_seed_mean
                # Recs[normal_idx] = Recs_seed_mean
            except:
                print('error')
        try:
            # print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\tlr:{}".format(
            #     opt.model, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
            #     np.mean(list(APs.values())), np.std(list(APs.values())),np.mean(list(Pres.values())), np.std(list(Pres.values())),np.mean(list(Recs.values())), np.std(list(Recs.values())), np.max(list(MAX_EPOCHs.values())),opt.lr
            # ), file=file2print)

            print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\tlr:{}".format(
                opt.model, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
                np.mean(list(APs.values())), np.std(list(APs.values())),  opt.lr
            ), file=file2print)
        except:
            print('error')

        # print("################## {} ###################".format(dataset_name), file=file2print)
        # print("AUCs={} \n APs={}".format(AUCs, APs), file=file2print)
        # print("AUC:\n mean={}, std={}".format(round(np.mean(list(AUCs.values())), 4),
        #                                       round(np.std(list(AUCs.values())), 4)), file=file2print)
        # print("AP:\n mean={}, std={}".format(round(np.mean(list(APs.values())), 4),
        #                                      round(np.std(list(APs.values())), 4)), file=file2print)
        #
        # print("@" * 30, file=file2print)

        file2print.flush()






