import json
from datetime import datetime
import torch.nn as nn

from lib_for_SLMR.args import get_parser
from lib_for_SLMR.utils import *
#from lib_for_SLMR.prediction import Predictor
from lib_for_SLMR.training import Trainer
#from m_cnn import Res2NetBottleneck
#from m_cnn_pool import Res2NetBottleneck
#from m_cnn_pool_se import Res2NetBottleneck
#from m_cnn_pool_se_split import Res2NetBottleneck
from lib_for_SLMR.se_in_gru_zhiqian import Res2NetBottleneck

#MultiModels
from options_SLMR import Options
from dataloader.UCR_dataloader import load_data
import random
device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
#torch.cuda.set_per_process_memory_fraction(0.05, 1)

opt = Options().parse()

id = datetime.now().strftime("%d%m%Y_%H%M%S")


dataset = opt.dataset
window_size = opt.lookback
spec_res = opt.spec_res
normalize = opt.normalize
n_epochs = opt.epochs
batch_size = opt.batchsize
init_lr = opt.init_lr
val_split = opt.val_split
shuffle_dataset = opt.shuffle_dataset
# use_cuda = opt.use_cuda
print_every = opt.print_every
log_tensorboard = opt.log_tensorboard
group_index = opt.group[0]
index = opt.group[2:]
args_summary = str(opt.__dict__)
print(args_summary)
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1

DATASETS_NAME={
    #'YXD':1,
    'PTB-XL':1,
    'YXD':1
}
SEEDS=[
    1,2,3
]

if __name__ == '__main__':
    opt.model = "SLMR"
    results_dir='./logs2'
    #results_dir = '/home/tcr/storage/HXH/MultiModal/experiments/ecg/log10'
    #results_dir = '/home/g817_u2/XunHua/MultiModal/experiments/ecg/log10'

    opt.outf = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
    file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')

    import datetime
    print(datetime.datetime.now())
    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)

    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
    file2print.flush()
    file2print_detail.flush()

    for dataset_name in list(DATASETS_NAME.keys()):

        AUCs={}
        APs={}
        F1={}
        MAX_EPOCHs = {}

        for normal_idx in range(DATASETS_NAME[dataset_name]):
            normal_idx=1



            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))

            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            f1_seed={}
            for seed in SEEDS:
                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.backends.cudnn.deterministic = True
                
                opt.seed = seed

                opt.normal_idx = normal_idx
                dataloader, opt.isize, opt.signal_length = load_data(opt,dataset_name, None, True)
                opt.dataset = dataset_name

                #print("[INFO] Class Distribution: {}".format(class_stat))

                #

                opt.name = "%s/%s" % (opt.model, opt.dataset)
                expr_dir = os.path.join(opt.outf, opt.name, 'train')
                test_dir = os.path.join(opt.outf, opt.name, 'test')

                if not os.path.isdir(expr_dir):
                    os.makedirs(expr_dir)
                if not os.path.isdir(test_dir):
                    os.makedirs(test_dir)

                output_path = f'output/SLMR/'
                log_dir = f'{output_path}/logs'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                save_path = f"{output_path}/{id}"


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

                model = Res2NetBottleneck(
                    inplanes=opt.nc,
                    planes=150,
                    out_dim=opt.nc,
                    pre_len=opt.pre_len,
                    window=opt.isize,
                    gru_hid_dim=opt.gru_hid_dim,
                    gru_n_layers=opt.gru_n_layers,
                    fc_hid_dim=opt.fc_hid_dim,
                    fc_n_layers = opt.fc_n_layers,
                    recon_hid_dim=opt.recon_hid_dim,
                    recon_n_layers=opt.recon_n_layers,
                    downsample=True,
                    dropout=opt.dropout,
                    device=device,
                    stride=1,
                    scales=5,
                    groups=6,
                    se=False,
                    norm_layer=None)

                # print(model)   
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
                forecast_criterion = nn.MSELoss()
                recon_criterion = nn.MSELoss()
                opt.istest=True


                if opt.istest==True:
                    model.load_state_dict(torch.load('./Model_CheckPoints2/{}_{}_{}_{}_{}.pkl'.format(opt.model, opt.dataset,
                                                                               opt.normal_idx,
                                                                                opt.seed, opt.lr), map_location='cpu'))
                    trainer = Trainer(
                        model,
                        optimizer,
                        opt.isize,
                        opt.nc,
                        None,
                        n_epochs,
                        batch_size,
                        init_lr,
                        forecast_criterion,
                        recon_criterion,
                        save_path,
                        log_dir,
                        print_every,
                        log_tensorboard,
                        args_summary,
                        device
                    )


                    ap_test, auc_test,f1_test, epoch_max_point=trainer.predict(opt,dataloader=dataloader["test"])
                else:
                    trainer = Trainer(
                        model,
                        optimizer,
                        opt.isize,
                        opt.nc,
                        None,
                        n_epochs,
                        batch_size,
                        init_lr,
                        forecast_criterion,
                        recon_criterion,
                        save_path,
                        log_dir,
                        print_every,
                        log_tensorboard,
                        args_summary,
                        device
                    )


                    ap_test, auc_test,f1_test, epoch_max_point = trainer.fit(opt, dataloader['train'], dataloader['val'], dataloader['test'])
    

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                f1_seed[seed] = f1_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

                # End For

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)
            f1_seed_mean = round(np.mean(list(f1_seed.values())), 4)
            f1_seed_std = round(np.std(list(f1_seed.values())), 4)

            print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{}\t F1={}+{} \t MAX_EPOCHs={}".format(
                dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,f1_seed_mean,f1_seed_std, MAX_EPOCHs_seed))

            print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                opt.model, dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,f1_seed_mean,f1_seed_std,
                MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            F1[normal_idx]=f1_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
            opt.model, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
            np.mean(list(APs.values())), np.std(list(APs.values())),np.mean(list(F1.values())), np.std(list(F1.values())), np.max(list(MAX_EPOCHs.values()))
        ), file=file2print)

        # print("################## {} ###################".format(dataset_name), file=file2print)
        # print("Pres={} \n Recalls={}".format(Pres, Recalls), file=file2print)
        # print("AUC:\n mean={}, std={}".format(round(np.mean(list(Pres.values())), 4),
        #                                       round(np.std(list(Pres.values())), 4)), file=file2print)
        # print("AP:\n mean={}, std={}".format(round(np.mean(list(Recalls.values())), 4),
        #                                      round(np.std(list(Recalls.values())), 4)), file=file2print)
        #
        # print("@" * 30, file=file2print)

        file2print.flush()

