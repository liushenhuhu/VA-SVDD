import torch
import logging
import random
import numpy as np

from lib_for_DeepSVDD.utils.config import Config
from model.DeepSVDD  import DeepSVDD
# from deepSVDD_RP import DeepSVDD_RP
# from deepSVDD_CAT import DeepSVDD_CAT
# from deepSVDD_FFT_SIN import DeepSVDD_FFT_SIN
from lib_for_DeepSVDD.datasets.main import load_dataset
import os
import json
################################################################################
# Settings
################################################################################

#nohup /home/tcr/anaconda3/bin/python3.8  /home/tcr/storage/HXH/MultiModal/experiments/ecg/main_DeepSVDD.py >/dev/null 2>&1 &
#nohup /home/g817_u2/anaconda3/envs/torch1.4/bin/python3.6  /home/g817_u2/XunHua/MultiModal/experiments/ecg/main_DeepSVDD.py >/dev/null 2>&1 &



'''
python main.py 
    mnist 
    mnist_LeNet 
    ../log/mnist_test 
    ../data 
    --objective one-class 
    --lr 0.0001 
    --n_epochs 150 
    --lr_milestone 50 
    --batch_size 200 
    --weight_decay 0.5e-6 
    --pretrain True 
    --ae_lr 0.0001 
    --ae_n_epochs 150 
    --ae_lr_milestone 50 
    --ae_batch_size 200 
    --ae_weight_decay 0.5e-3 
    --normal_class 3;

'''





def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, checkpoint_pth, seed):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataloader to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/DeepSVDD/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, net_name, normal_class, seed)
    #dataloader, opt.isize, opt.signal_length = load_data(opt, dataset_name, None, True)
    # Initialize DeepSVDD model and set neural network \phi
    if net_name in ['ucr_OSCNN_RP', 'ucr_OSCNN_FFT', 'ucr_CNN_FFT']:
        deep_SVDD = DeepSVDD_RP(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name, dataset.signal_length, dataset.rp_size)

    elif net_name in ['ucr_CNN_FFT_CAT']:
        deep_SVDD = DeepSVDD_CAT(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name, dataset.signal_length, dataset.rp_size)

    elif net_name in ['ucr_CNN_FFT_SIN']:
        deep_SVDD = DeepSVDD_FFT_SIN(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name, dataset.rp_size)


    else:
        deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name, dataset.signal_length)

    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataloader (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader,
                           signal_length=dataset.signal_length)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataloader
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    checkpoint_pth=checkpoint_pth)

    #return deep_SVDD.trainer.test_pre, deep_SVDD.trainer.test_rec, deep_SVDD.trainer.test_f1,deep_SVDD.trainer.max_epoch
    return deep_SVDD.trainer.test_auc, deep_SVDD.trainer.test_ap, deep_SVDD.trainer.max_epoch

    # # Test model
    # deep_SVDD.test(dataloader, device=device, n_jobs_dataloader=n_jobs_dataloader)
    #
    # # Plot most anomalous and most normal (within-class) test samples
    # indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
    #
    # if dataset_name in ('mnist', 'cifar10'):
    #
    #     if dataset_name == 'mnist':
    #         X_normals = dataloader.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
    #         X_outliers = dataloader.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)
    #
    #     if dataset_name == 'cifar10':
    #         X_normals = torch.tensor(np.transpose(dataloader.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
    #         X_outliers = torch.tensor(np.transpose(dataloader.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))
    #
    #     plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
    #     plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)
    #
    # # Save results, model, and configuration
    # deep_SVDD.save_results(export_json=xp_path + '/results.json')
    # deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    # cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    from options import Options

    # DATA_ROOT = '../datasets/UCRArchive_2018'
    # DATA_ROOT = '../data'
    DATASETS_NAME = {
        'PTB-XL':1
    }
    SEEDS = [
        1,2,3,4,5
    ]

    opt = Options().parse()
    device = torch.device("cuda:1" if
                          torch.cuda.is_available() else "cpu")

    results_dir = './log1'
    #results_dir = '/home/tcr/storage/HXH/MultiModal/experiments/ecg/log1'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file2print = open('{}/results_DeepSVDD_{}.log'.format(results_dir, opt.net_name), 'a+')
    file2print_detail = open('{}/results_DeepSVDD_{}_detail.log'.format(results_dir, opt.net_name), 'a+')

    import datetime

    print(datetime.datetime.now())
    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)
    # print("Model\tDataset\tNormal_Label\tPre_mean\tPre_std\tRec_mean\tRec_std\F1_mean\tF1_std\tMax_Epoch", file=file2print_detail)
    #
    # print("Model\tDataset\tNormal_Label\tPre_mean\tPre_std\tRec_mean\tRec_std\F1_mean\tF1_std\tMax_Epoch")
    # print("Model\tDataset\tNormal_Label\tPre_mean\tPre_std\tRec_mean\tRec_std\F1_mean\tF1_std\tMax_Epoch", file=file2print)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)

    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)



    file2print.flush()
    file2print_detail.flush()

    for dataset_name in list(DATASETS_NAME.keys()):

        AUCs = {}
        APs = {}
        MAX_EPOCHs = {}
        Pres = {}
        Recs = {}
        F1s = {}

        for normal_class in range(DATASETS_NAME[dataset_name]):
            # for normal_class in [0]:
            # normal_class = 1

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_class))

            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            Pres_seed = {}
            Recs_seed = {}
            F1_seed = {}


            for seed in SEEDS:
                np.random.seed(seed)
               # opt.seed = seed ** 2
                chk_dir = '{}/{}/{}/{}/{}'.format(results_dir,'DeepSVDD', dataset_name, normal_class, seed)
                if not os.path.exists(chk_dir):
                    os.makedirs(chk_dir)

                #checkpoint_pth = '/home/tcr/storage/HXH/MultiModal/log'
                checkpoint_pth = '../log'

                #pre_test, rec_test, f1_test, epoch_max_point = \
                auc_test, ap_test, epoch_max_point = \
                    main(dataset_name, opt.net_name, results_dir, opt.data_UCR, opt.load_config, opt.load_model,
                         opt.objective, opt.nu, device, opt.optimizer_name, opt.lr, opt.n_epochs,
                         opt.lr_milestone,
                         opt.batchsize, opt.weight_decay,
                         opt.pretrain, opt.ae_optimizer_name, opt.ae_lr, opt.ae_n_epochs, opt.ae_lr_milestone,
                         opt.ae_batch_size, opt.ae_weight_decay,
                         opt.n_jobs_dataloader, normal_class, checkpoint_pth,seed)

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                MAX_EPOCHs_seed[seed] = epoch_max_point
                # Pres_seed[seed] = pre_test
                # Recs_seed[seed] = rec_test
                # F1_seed[seed] = f1_test
                # MAX_EPOCHs_seed[seed] = epoch_max_point


                # End For

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)
            # Pres_seed_mean = round(np.mean(list(Pres_seed.values())), 4)
            # Pres_seed_std = round(np.std(list(Pres_seed.values())), 4)
            # Recs_seed_mean = round(np.mean(list(Recs_seed.values())), 4)
            # Recs_seed_std = round(np.std(list(Recs_seed.values())), 4)
            #
            # F1_seed_mean = round(np.mean(list(F1_seed.values())), 4)
            # F1_seed_std = round(np.std(list(F1_seed.values())), 4)


            # print("Dataset: {} \t Normal Label: {} \t Pres={}+{}\t Recs={}+{}\t  F1s={}+{} \tMAX_EPOCHs={}".format(
            #     dataset_name, normal_class, Pres_seed_mean, Pres_seed_std, Recs_seed_mean, Recs_seed_std, F1_seed_mean,F1_seed_std, MAX_EPOCHs_seed))
            #
            # print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
            #     opt.model, dataset_name, normal_class, Pres_seed_mean, Pres_seed_std, Recs_seed_mean, Recs_seed_std,F1_seed_mean, F1_seed_std,MAX_EPOCHs_seed_max), file=file2print_detail)
            # file2print_detail.flush()


            print("Dataset: {} \t Normal Label: {} \t AUC={}+{}\t AP={}+{}\t MAX_EPOCHs={}".format(
                dataset_name, normal_class, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed))

            print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                opt.net_name, dataset_name, normal_class, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,MAX_EPOCHs_seed_max), file=file2print_detail)
            file2print_detail.flush()


            AUCs[normal_class] = AUCs_seed_mean
            APs[normal_class] = APs_seed_mean
            MAX_EPOCHs[normal_class] = MAX_EPOCHs_seed_max

            # Pres[normal_class] = Pres_seed_mean
            # Recs[normal_class] = Recs_seed_mean
            # F1s[normal_class] = F1_seed_mean

        # print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format( opt.model, dataset_name, np.mean(list(Pres.values())), np.std(list(Pres.values())),
        #     np.mean(list(Recs.values())), np.std(list(Recs.values())), np.mean(list(F1s.values())), np.std(list(F1s.values())), np.max(list(MAX_EPOCHs.values()))), file=file2print)


        print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format( opt.net_name, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
            np.mean(list(APs.values())), np.std(list(APs.values())) ,np.max(list(MAX_EPOCHs.values()))), file=file2print)


        # print("################## {} ###################".format(dataset_name), file=file2print)
        # print("AUCs={} \n APs={}".format(AUCs, APs), file=file2print)
        # print("AUC:\n mean={}, std={}".format(round(np.mean(list(AUCs.values())), 4),
        #                                       round(np.std(list(AUCs.values())), 4)), file=file2print)
        # print("AP:\n mean={}, std={}".format(round(np.mean(list(APs.values())), 4),
        #                                      round(np.std(list(APs.values())), 4)), file=file2print)
        #
        # print("@" * 30, file=file2print)

        file2print.flush()
