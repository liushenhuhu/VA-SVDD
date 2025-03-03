#/home/tcr/anaconda3/envs/tf1.14/bin/python3.6 -u /home/tcr/storage/HXH/MultiModal/experiments/ecg/main.py

import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        '''
        ElectricDevices
        EthanolLevel
        MiddlePhalanxOutlineAgeGroup
        MixedShapesRegularTrain
        Symbols
        TwoPatterns
        '''

        ##
        # Base
        #self.parser.add_argument('--dataloader', default='ACSF1', help='run dataloader')
        self.parser.add_argument('--data',default='UCR',help='ECG or UCR')
        self.parser.add_argument('--dataloader', default='ElectricDevices', help='run dataloader')
        self.parser.add_argument('--data_ECG', default='../datasets/ECG', help='path to ECG')
        #self.parser.add_argument('--data_UCR',default='../datasets/UCRArchive_2018',help='path to UCRdataset')
        self.parser.add_argument('--data_UCR',default='/home/huangxunhua/UCRArchive_2018',help='path to UCRdataset')
        # self.parser.add_argument('--data_UCR',default='/home/g817_u2/XunHua/MultiModal/experiments/datasets/UCRArchive_2018',help='path to UCRdataset')
        # self.parser.add_argument('--data_DECG',default='/home/g817_u2/XunHua/MultiModal/experiments/datasets/DECG',help='path to UCRdataset')

        self.parser.add_argument('--data_CPSC',default='/home/tcr/storage/HXH/MultiModal/experiments/datasets/CSPC_ECG')
        self.parser.add_argument('--data_ZZU_MI',default='/home/tcr/storage/HXH/MultiModal/experiments/datasets/zzu_MI')
        self.parser.add_argument('--data_DECG',default='/home/tcr/storage/HXH/MultiModal/experiments/datasets/DECG')

        self.parser.add_argument('--isMask', type=bool, default=True)
        self.parser.add_argument('--Maskwhere', default='random',choices=['Q','ST','T','random'])
        self.parser.add_argument('--isMasklead', type=bool, default=True)
        self.parser.add_argument('--Maskleadnum', type=int, default=3)
        self.parser.add_argument('--Maskleadwhere', type=int, default=0)
        self.parser.add_argument('--isInterpret', type=bool, default=False)
        self.parser.add_argument('--isVisualize', type=bool, default=True)

        #/home/tcr/HXH/MultiModal/experiments/ecg

        self.parser.add_argument('--data_MI',default='../datasets/MI/')
        self.parser.add_argument('--data_EEG',default='../datasets/EEG/')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
        self.parser.add_argument('--isize', type=int, default=166, help='input sequence size.')

        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

        self.parser.add_argument('--nc', type=int, default=2, help='input sequence channels')
        self.parser.add_argument('--nz', type=int, default=4, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)
        #[AE_CNN_4indicator,BeatGAN_4indicator,Ganomaly_4indicator,USAD_4indicator,Ganomaly_UNET]
        self.parser.add_argument('--model', type=str, default='Ganomaly_4indicator',
                                 choices=['AE_CNN_4indicator','BeatGAN_4indicator','USAD_4indicator','Ganomaly_4indicator','DeepSVDD'])
        self.parser.add_argument('--FilterType', type=str, default='EM-KF', choices=['Savitzky', 'Wiener', 'Kalman','EM-KF'])

        self.parser.add_argument('--Snr', type=float, default=5, help='the snr of noisy')
        self.parser.add_argument('--normal_idx', type=int, default=0, help='the label index of normaly')
        self.parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
        self.parser.add_argument('--early_stop', type=int, default=200)
        self.parser.add_argument('--pre_niter', type=int, default=200)


        self.parser.add_argument('--eta', type=float, default=0.8, help='Raw signal reconstruction error weights.')
        self.parser.add_argument('--theta', type=float, default=0.2, help='WaveLet signal reconstruction error weights.')

        self.parser.add_argument('--outf', default='./output', help='output folder')

        self.parser.add_argument('--eps', type=float, default=1e-10, help='number of GPUs to use')
        self.parser.add_argument('--pool', type=str, default='max', help='number of GPUs to use')
        self.parser.add_argument('--normalize', type=bool, default='True')
        self.parser.add_argument('--seed', type=int, default=0)

        # ganomaly
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--decoder_isize', type=int, default=32, help='input sequence size.')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--w_mask', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')



        #######UNET
        self.parser.add_argument("--experiment_name", type=str,
                            default="MultiCGAN_PTBXL", help="name of the experiment")
        self.parser.add_argument("--shuffle_training", type=bool,
                            default=True, help="shuffle training")
        self.parser.add_argument("--shuffle_testing", type=bool,
                            default=False, help="shuffle testing")
        self.parser.add_argument("--is_eval", type=bool,
                            default=False, help="evaluation mode")
        self.parser.add_argument("--from_ppg", type=bool, default=True,
                            help="reconstruct from ppg")
        self.parser.add_argument("--peaks_only", type=bool, default=False,
                            help="L2 loss on peaks only")
        self.parser.add_argument("--epoch", type=int, default=0,
                            help="epoch to start training from")
        # self.parser.add_argument("--n_epochs", type=int, default=15000,
        #                     help="number of epochs of training")
        # self.parser.add_argument("--batch_size", type=int, default=1024,
        #                     help="size of the batches")
        # self.parser.add_argument("--lr", type=float, default=0.0002,
        #                     help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--lambda_gp", type=float, default=10,
                            help="Loss weight for gradient penalty")
        self.parser.add_argument("--ncritic", type=int, default=3,
                            help=" number of iterations of the critic per generator iteration")
        # self.parser.add_argument("--n_cpu", type=int, default=4,
        #                     help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--signal_length", type=int,
                            default=1000, help="size of the signal")
        self.parser.add_argument("--checkpoint_interval", type=int,
                            default=500, help="interval between model checkpoints")









        

        ### DeepSVDD###
        self.parser.add_argument('--dataset_name', default='cpsc2021',choices=['cpsc2021','ptbxl'],help='run dataloader')
        self.parser.add_argument('--net_name', default='ucr_CNN',
                                 choices=['ucr_CNN', 'ucr_OSCNN',
                                          'ucr_OSCNN_RP', 'ucr_OSCNN_FFT', 'ucr_CNN_FFT',
                                          'cifar10_LeNet', 'ucr_CNN_FFT_CAT', 'ucr_CNN_FFT_SIN', 'AE_PyG'],
                                 help='Model name.')
        self.parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs to train.')

        #self.parser.add_argument('--xp_path', default='./log1', help='run dataloader')
        # self.parser.add_argument('--data_path', default='./datasets/UCRArchive_2018', help='Dataset path')

        #self.parser.add_argument('--data_path', default='../datasets/UCRArchive_2018', help='Dataset path')
        # self.parser.add_argument('--data_path', default='./datasets/MI', help='Dataset path')
        #self.parser.add_argument('--data_path', default='../datasets/DECG', help='Dataset path')


        self.parser.add_argument('--load_config', default=None, help='Config JSON-file path (default: None).')
        self.parser.add_argument('--load_model', default=None, help='Model file path (default: None).')
        self.parser.add_argument('--objective', default='one-class', choices=['one-class', 'soft-boundary'],
                                 help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
        self.parser.add_argument('--nu', type=float, default=0.1,
                                 help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
        # self.parser.add_argument('--device', type=str, default='cuda:0',
        #                          #help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
        #self.parser.add_argument('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
        self.parser.add_argument('--optimizer_name', choices=['adam'], default='adam',
                                 help='Name of the optimizer to use for Deep SVDD network training.')
        #self.parser.add_argument('--lr', type=float, default=0.01,
                                 #help='Initial learning rate for Deep SVDD network training. Default=0.001')
        self.parser.add_argument('--lr_milestone', type=list, default=[0],
                                 help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
        #self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for mini-batch training.')
        self.parser.add_argument('--weight_decay', type=float, default=1e-6,
                                 help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
        self.parser.add_argument('--pretrain', type=bool, default=False,
                                 help='Pretrain neural network parameters via autoencoder.')
        self.parser.add_argument('--ae_optimizer_name', choices=['adam'], default='adam',
                                 help='Name of the optimizer to use for autoencoder pretraining.')
        self.parser.add_argument('--ae_lr', type=float, default=0.0001,
                                 help='Initial learning rate for autoencoder pretraining. Default=0.001')
        self.parser.add_argument('--ae_n_epochs', type=int, default=400, help='Number of epochs to train autoencoder.')
        self.parser.add_argument('--ae_lr_milestone', type=list, default=[0],
                                 help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
        self.parser.add_argument('--ae_batch_size', type=int, default=64,
                                 help='Batch size for mini-batch autoencoder training.')
        self.parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
                                 help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
        self.parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                                 help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
        self.parser.add_argument('--normal_class', type=int, default=0,
                                 help='Specify the normal class of the dataloader (all other classes are considered anomalous).')

        #####GOAD#####

        # args参数

        self.parser.add_argument('--n_rots', default=256, type=int)
        self.parser.add_argument('--d_out', default=32, type=int)
        self.parser.add_argument('--exp', default='affine', type=str)
        self.parser.add_argument('--c_pr', default=0, type=int)
        self.parser.add_argument('--m', default=1, type=float)
        self.parser.add_argument('--lmbda', default=0.1, type=float)
        #self.parser.add_argument('--eps', default=0, type=float)


        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')

        #self.parser.add_argument('--signal_length',type=list,default=[48],help='the length of wavelet signal')
        ## Test
        self.parser.add_argument('--istest',action='store_true', default=False, help='train model_eeg or test model_eeg')
        self.parser.add_argument('--threshold', type=float, default=0.05, help='threshold score for anomaly')

        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk


        # self.opt.name = "%s/%s" % (self.opt.model_eeg, self.opt.dataloader)
        # expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        # test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        #
        # if not os.path.isdir(expr_dir):
        #     os.makedirs(expr_dir)
        # if not os.path.isdir(test_dir):
        #     os.makedirs(test_dir)
        #
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        return self.opt