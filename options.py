import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and vf-fake-promax options
    """

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


        self.parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--isize', type=int, default=166, help='input sequence size.')

        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        # self.parser.add_argument('--is_load_train', type=bool, default=True, help='is load method')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')


        self.parser.add_argument('--nc', type=int, default=1, help='input sequence channels')
        self.parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)

        ##
        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=50,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')
        self.parser.add_argument('--signal_length', type=list, default=[48], help='the length of wavelet signal')
        self.parser.add_argument('--spetrum_wide', type=list, default=[63], help='63,the length of wavelet signal')
        self.parser.add_argument('--spetrum_high', type=list, default=[17], help='17,the length of wavelet signal')
        self.parser.add_argument('--alpha', type=int, default=0.1, help='alpha in loss function')

        self.parser.add_argument('--normal_idx', type=int, default=0, help='the label index of normaly')
        self.parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
        self.parser.add_argument('--early_stop', type=int, default=200)

        # 附加参数
        self.parser.add_argument('--seed', type=int, default=1)
        self.parser.add_argument('--noisetest', type=bool, default=False)

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

        return self.opt
