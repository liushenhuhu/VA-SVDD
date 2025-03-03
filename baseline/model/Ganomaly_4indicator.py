"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from baseline.lib_for_Ganomaly.networks_1D import NetG, NetD, weights_init
from baseline.lib_for_Ganomaly.visualizer import Visualizer
from baseline.lib_for_Ganomaly.loss import l2_loss
#from lib_for_Ganomaly.evaluate import evaluate
from baseline.model.metric import evaluate
"""原始Ganomaly  单通道"""
from baseline.utils_plot.draw_utils import plot_hist,plot_tsne_sns
class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader, device):
        ##
        # Seed for deterministic behavior
        #self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        #self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        # self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        # self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = device

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[1].size()).copy_(input[1])
            self.realecg.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[2].size()).copy_(input[2])
            self.label.resize_(input[2].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[1].size()).copy_(input[1])
                self.fixed_realecg.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.model, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                #                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d.   G_loss:  %d   D_loss:  %d" % (self.opt.model, self.epoch + 1, self.opt.niter,self.err_g,self.err_d))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_result = 0
        best_ap = 0
        best_auc = 0
        best_rec = 0
        best_pre = 0
        best_f1 = 0
        best_result_epoch = 0
        early_stop_epoch = 0
        early_stop_auc = 0
        early_stop_results = 0

        auc_test = 0
        ap_test = 0

        # Train for niter epochs.
        print(">> Training model %s" % self.opt.model)
        for self.epoch in range(0, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()

            ap_val, auc_val, pre_val, rec_val, f1 = self.validate()




            if (pre_val + rec_val) > best_result:

                best_result = pre_val+ rec_val
                best_pre = pre_val
                best_rec = rec_val
                best_f1 = f1

                best_result_epoch = self.epoch

                ap_test, auc_test ,Pre_test, Rec_test, F1_test = self.test()



                if self.epoch == 1:
                    early_stop_results = (Pre_test + Rec_test)

            if (Pre_test + Rec_test ) <= early_stop_results:
                early_stop_epoch = early_stop_epoch + 1
            else:
                early_stop_epoch = 0
                early_stop_results = (Pre_test + Rec_test)
                torch.save(self.netg.state_dict(),
                       '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(self.opt.model, self.opt.dataset, self.opt.normal_idx,
                                                                   self.opt.seed))

            if early_stop_epoch == self.opt.early_stop:
                break


            print("EPOCH [{}] Pre:{:.4f} \t  Rec:{:.4f} \t  F1:{:.4f} \t BEST VAL pre:{:.4f} \t  VAL_rec:{:.4f}\t  VAL_f1:{:.4f}  \t in epoch[{}] \t TEST  Pre:{:.4f} \t Rec:{:.4f}\t  F1:{:.4f}\t EarlyStop [{}] \t".format(
                    self.epoch, pre_val, rec_val, f1, best_pre, best_rec, best_f1,best_result_epoch, Pre_test, Rec_test , F1_test, early_stop_epoch))

        print(">> Training model %s.[Done]" % self.name)


        return ap_test, auc_test, Pre_test, Rec_test,best_result_epoch


    ##VAL########
    def validate(self):
        """ VAL GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the VAL set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'VAL'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['val'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['val'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['val'].dataset), self.opt.nz),
                                        dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['val'].dataset), self.opt.nz),
                                        dtype=torch.float32,
                                        device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['val'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'val', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                    torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            #auc, ap, Pre, Rec, F1 = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            ap, auc, pre, rec, F1= evaluate(self.gt_labels.cpu(), np.array(self.an_scores.cpu()))

            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'val':
                counter_ratio = float(epoch_iter) / len(self.dataloader['val'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return ap, auc, pre, rec, F1

##



    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            #auc, ap, Pre, Rec,F1 = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            ap, auc,pre, rec, F1 = evaluate(self.gt_labels.cpu(), np.array(self.an_scores.cpu()))
            if self.opt.istest==True:
                plot_hist(self.an_scores.cpu().numpy(), self.gt_labels.cpu().numpy(), self.opt, 'Ganomaly_4indicator',display=False)
                plot_tsne_sns(self.latent_i.cpu().numpy(), self.gt_labels.cpu().numpy(), self.opt,'Ganomaly_4indicator_latent_i', display=False)
                plot_tsne_sns(self.latent_o.cpu().numpy(), self.gt_labels.cpu().numpy(), self.opt,'Ganomaly_4indicator_latent_o', display=False)

            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return ap, auc, pre, rec, F1


##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, dataloader, device):
        super(Ganomaly, self).__init__(opt, dataloader, device)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.device = device

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.realecg = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        self.fixed_realecg = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        #if self.opt.isTrain:
        self.netg.train()
        self.netd.train()
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

        #latent1, latent2, latent3, latent4, gen_ecg1, gen_ecg2,

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.realecg)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        a = self.netd(self.input)
        self.err_g_adv = self.l_adv(self.netd(self.realecg)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.realecg)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

        #self.l_adv = l2_loss
        # self.l_con = nn.L1Loss()
        # self.l_enc = l2_loss
        # self.l_bce = nn.BCELoss()


    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
