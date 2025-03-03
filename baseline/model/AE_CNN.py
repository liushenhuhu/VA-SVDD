import time,os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .network import weights_init
from metric import evaluate
import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 16})

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from Dilate_Loss.dilate_loss import dilate_loss
dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))
from .plotUtil import  save_ts_heatmap_2D, save_ts_heatmap_1D, save_ts_heatmap_1D_Batch

# from TSNE.TSNE import do_tsne, do_tsne_sns
# from TSNE.Do_Hist import do_hist


#from KalmanNet import KalmanNetNN

# class Encoder(nn.Module):
#     def __init__(self, ngpu, opt, out_z):
#         super(Encoder, self).__init__()
#         self.ngpu = ngpu
#
#         self.cov1 = nn.Sequential(
#             # input is (nc) x 320   (1,320)
#             nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),   # (1,32,4) --> (32,)
#             nn.BatchNorm1d(opt.ndf),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#
#         self.cov2 = nn.Sequential(
#             # state size. (ndf) x 160
#             nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),   # 32,64
#             nn.BatchNorm1d(opt.ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             #state size. (ndf*2) x 80
#         )
#
#         self.cov3 = nn.Sequential(
#
#             nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  #64 ,128
#             nn.BatchNorm1d(opt.ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#
#         self.cov4 = nn.Sequential(
#
#             nn.Conv1d(opt.ndf * 4, out_z, 2, 1, 0, bias=False),  #128,50
#             #nn.Conv1d(opt.ndf * 16, out_z, 10, 1, 0, bias=False),
#             # state size. (nz) x 1
#             nn.AdaptiveAvgPool1d(1)
#         )
#
#
#     def forward(self, input):   #(1,1600)
#
#         out = self.cov1(input)  # (32,800)
#         out = self.cov2(out)  # (64,400)
#         out = self.cov3(out)  # (128,200)
#         out = self.cov4(out)  # (40,1)
#
#         return out


class Encoder(nn.Module):
    def __init__(self, ngpu, opt, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320   (1,320)
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),   # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 160
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),   # 32,64
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. (ndf*2) x 80

            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  #64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 40
            # nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(opt.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # state size. (ndf*8) x 20
            # nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(opt.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(opt.ndf * 4, out_z, 2, 1, 0, bias=False),  #128,50
            #nn.Conv1d(opt.ndf * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class Decoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),  # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False), # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False), # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False), # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            #nn.BatchNorm1d(opt.nc),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, opt.isize), # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
            #nn.BatchNorm1d(200),
            #nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z):#(8,1)
        out = self.deconv1(z)  #(128,10)
        out = self.deconv2(out) #(64,20)
        out = self.deconv3(out) #(32,40)
        out = self.deconv4(out) #(1,80)
        out = self.fc(out)

        return out

class AE_CNN(nn.Module):

    def __init__(self, opt):
        super(AE_CNN, self).__init__()
        opt.isize

        self.encoder1 = Encoder(opt.ngpu, opt, opt.nz)

        self.decoder = Decoder(opt.ngpu, opt)
        # self.signal_length = opt.signal_length

    def forward(self, x): # （64，1，320）
        # latent_i = self.encoder1(x[:, :, :self.signal_length[0]])  #（64，50，1）
        #x = x[:, np.newaxis, :]
        latent_i = self.encoder1(x)  #（64，50，1）

        gen_x = self.decoder(latent_i)  #（64，1，320）
        return gen_x, latent_i


class ModelTrainer(nn.Module):

    def __init__(self, opt, dataloader, device):
        super(ModelTrainer, self).__init__()
        self.niter=opt.niter
        self.dataset=opt.dataset
        self.model = opt.model
        self.outf=opt.outf
        self.normal_idx = opt.normal_idx
        self.seed = opt.seed

        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = AE_CNN(opt).to(device)
        self.G.apply(weights_init)

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.dilate_loss = dilate_loss

        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch=0

        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)

        self.time_step = torch.empty(size=(self.opt.batchsize,) , device=self.device)
        self.anno_step = torch.empty(size=(self.opt.batchsize,) , device=self.device)


        self.latent_i = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None
        self.train_finish = False

        self.gamma = 0.001
        self.alpha = 0.8

        self.fake_all = None
        self.input_all = None
        self.mse_mean = None

        self.tsne_pred = None
        self.tsne_true = None
        self.tsne_latent = None

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train AECNN.")
        start_time = time.time()

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
        th_test = 0
        F1_test = 0
        Pre_test = 0
        Rec_test = 0

        # with open(os.path.join(self.outf, self.model, self.dataloader, "val_info.txt"), "w") as f:
        #     for epoch in range(self.niter):
        #         self.cur_epoch+=1
        #
        #         # Train
        #         self.train_epoch()
        #             # Val
        #         ap, auc, th, pre, rec, f1 = self.validate()
        #
        #
        #         if (pre+rec) > best_result:
        #
        #             best_result = pre+rec
        #             best_pre = pre
        #             best_rec = rec
        #             best_f1 = f1
        #
        #
        #             best_result_epoch = self.cur_epoch
        #
        #
        #
        #             # Test
        #             ap_test, auc_test, th_test, Pre_test, Rec_test, F1_test = self.test()
        #
        #
        #         #         if epoch == 1:
        #         #         early_stop_auc = auc_test
        #         #
        #         # if auc_test <= early_stop_auc :
        #         #     early_stop_epoch = early_stop_epoch+1
        #         # else:
        #         #     early_stop_epoch = 0
        #         #     early_stop_auc = auc_test
        #         #
        #         # if early_stop_epoch == self.opt.early_stop:
        #         #
        #         #         break
        #
        #             if epoch == 1:
        #                 early_stop_results = (Pre_test+Rec_test)
        #
        #         if (Pre_test+Rec_test) <= early_stop_results :
        #             early_stop_epoch = early_stop_epoch+1
        #         else:
        #             early_stop_epoch = 0
        #             early_stop_results = (Pre_test+Rec_test)
        #
        #         if early_stop_epoch == self.opt.early_stop:
        #
        #                 break
        #
        #         f.write("EPOCH [{}] Pre:{:.4f} \t  Rec:{:.4f} \t  F1:{:.4f} \t BEST VAL pre:{:.4f} \t  VAL_rec:{:.4f}\t  VAL_f1:{:.4f}  \t in epoch[{}] \t TEST  Pre:{:.4f} \t Rec:{:.4f}\t  F1:{:.4f}\t EarlyStop [{}] \t".format(
        #             self.cur_epoch, pre, rec, f1, best_pre, best_rec, best_f1,best_result_epoch, Pre_test, Rec_test , F1_test, early_stop_epoch))
        #         print( "EPOCH [{}]  loss:{:.4f} \t Pre:{:.4f} \t  Rec:{:.4f} \t  F1:{:.4f} \t BEST VAL pre:{:.4f} \t  VAL_rec:{:.4f}\t  VAL_f1:{:.4f}  \t in epoch[{}] \t TEST  Pre:{:.4f} \t Rec:{:.4f}\t  F1:{:.4f}\t EarlyStop [{}] \t".format(
        #                 self.cur_epoch, self.err_g, pre, rec, f1, best_pre, best_rec, best_f1,best_result_epoch, Pre_test, Rec_test , F1_test, early_stop_epoch))
        #
        # self.train_hist['total_time'].append(time.time() - start_time)
        # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
        #                                                                 self.niter,
        #                                                                 self.train_hist['total_time'][0]))
        #
        # #save_ts_heatmap_1D(np.squeeze(self.input.cpu().numpy()),np.squeeze(self.fake.cpu().numpy()),'AE_CNN')
        # #save_ts_heatmap_1D_Batch(self.input.cpu().numpy(),self.fake.cpu().numpy(),'AE_CNN')
        #
        #
        #
        # return ap_test,auc_test,Pre_test,Rec_test, best_result_epoch


        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):

                self.cur_epoch+=1

                # Train
                self.train_epoch()

                input = self.input_all.detach().cpu().numpy()  #(960 1 1024)
                fake = self.fake_all.detach().cpu().numpy()

                input = input.reshape(-1)
                fake = fake.reshape(-1)

                self.mse_mean = np.mean(np.abs(fake - input))

                # Val
                ap, auc, th, f1 = self.validate()
                if auc > best_auc:

                    best_auc = auc
                    best_ap = ap

                    best_auc_epoch = self.cur_epoch


                    # Test
                    ap_test, auc_test, th_test, f1_test = self.test()



                    if epoch == 1:
                        early_stop_auc = auc_test

                if auc_test <= early_stop_auc:
                    early_stop_epoch = early_stop_epoch + 1
                else:
                    early_stop_epoch = 0
                    early_stop_auc = auc_test

                    #do_tsne_sns(self.tsne_latent, self.tsne_true, self.model, self.dataset, False, self.normal_idx,self.seed)


                    #torch.save(self.G.state_dict(), './Model_CheckPoint_3/{}_{}_{}_{}.pkl'.format(self.model, self.dataset, self.normal_idx,self.seed))

                if early_stop_epoch == self.opt.early_stop:
                    break

                f.write("EPOCH [{}] auc:{:.4f} \t  ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                    self.cur_epoch, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test ,early_stop_epoch))
                print( "EPOCH [{}]   \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch,  auc, ap, best_auc, best_ap, best_auc_epoch, auc_test ,ap_test, early_stop_epoch))


            # self.train_hist['total_time'].append(time.time() - start_time)
            # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
            #                                                                 self.niter,
            #                                                                 self.train_hist['total_time'][0]))

        return ap_test, auc_test, best_auc_epoch



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        epoch_iter = 0
        for i,data in enumerate(self.dataloader["train"], 0):
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()

            if i == 0 :

                self.input_all = self.input
                self.fake_all = self.fake
            else:
                self.input_all = torch.cat((self.input_all, self.input), 0)
                self.fake_all = torch.cat((self.fake_all, self.fake), 0)


            #
            #
            # if self.cur_epoch % 1000 == 0:
            #     save_ts_heatmap_1D(self.input.detach().cpu().numpy(), self.fake.detach().cpu().numpy(),
            #                        self.gt.detach().cpu().numpy(), 'AE_CNN_Train_{}_{}'.format(self.cur_epoch, i), self.mse_mean)
            #


            errors = self.get_errors()
            self.train_hist['G_loss'].append(errors["err_g"])
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])

            # self.time_step.resize_(input[2].size()).copy_(input[2])
            # self.anno_step.resize_(input[3].size()).copy_(input[3])

            # fixed input for view
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    def optimize(self):
        self.update_netg()

    def update_netg(self):
        self.G.zero_grad()
        self.fake, self.latent_i = self.G(self.input)   #G的生成结果
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x

        # self.fake = self.fake.reshape(self.fake.shape[0],self.fake.shape[2],self.fake.shape[1])
        # self.input = self.input.reshape(self.input.shape[0], self.input.shape[2], self.input.shape[1])
        #
        # self.err_g_rec = self.dilate_loss(self.fake, self.input, self.alpha, self.gamma, self.device)


        self.err_g = self.err_g_rec
        self.err_g.backward()
        self.optimizerG.step()



    def get_errors(self):

        errors = {
                    'err_g': self.err_g.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }


        return errors

    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]

        return  self.fixed_input.cpu().data.numpy(),fake.cpu().data.numpy()

    # def test(self):
    #     '''
    #     test by auc value
    #     :return: auc
    #     '''
    #     y_true, y_pred = self.predict(self.dataloader["test"])
    #     rocprc, rocauc, best_th,  Pre, Rec, f1 = evaluate(y_true, y_pred)
    #     return rocprc, rocauc, best_th, Pre, Rec, f1
    #
    # def validate(self):
    #     '''
    #     validate by auc value
    #     :return: auc
    #     '''
    #     y_true, y_pred = self.predict(self.dataloader["val"])
    #     rocprc, rocauc, best_th , Pre, Rec, f1= evaluate(y_true, y_pred)
    #     return rocprc, rocauc, best_th,  Pre, Rec, f1

    def test(self):
        '''
        test by auc value
        :return: auc
        '''

        y_true, y_pred = self.predict(self.dataloader["test"])
        rocprc, rocauc, best_th, best_f1 = evaluate(y_true, y_pred)
        return rocprc, rocauc, best_th, best_f1


    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["val"])
        rocprc, rocauc, best_th, best_f1 = evaluate(y_true, y_pred)
        return rocprc, rocauc, best_th, best_f1


    def predict(self,dataloader_,scale=True):

        #self.G.eval()
        with torch.no_grad():
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)

                # #if self.opt.istest:
                #
                # if self.cur_epoch % 1000 == 0:
                #
                #     save_ts_heatmap_1D(self.input.detach().cpu().numpy(), self.fake.detach().cpu().numpy(),
                #                    self.gt.detach().cpu().numpy(), 'AE_CNN_Test_{}_{}'.format(self.cur_epoch, i), self.mse_mean)
                #
                error = torch.mean( torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),dim=1)  #(64)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            self.tsne_latent = self.latent_i.cpu().numpy()
            self.tsne_true = y_true
            self.tsne_pred = y_pred




            return y_true, y_pred

    def test_time(self):
        self.G.eval()
        self.D.eval()
        size=self.dataloader["test"].dataset.__len__()
        start=time.time()

        for i, (data_x,data_y) in enumerate(self.dataloader["test"], 0):
            input_x=data_x
            for j in range(input_x.shape[0]):
                input_x_=input_x[j].view(1,input_x.shape[1],input_x.shape[2]).to(self.device)
                gen_x,_ = self.G(input_x_)

                error = torch.mean(
                    torch.pow((input_x_.view(input_x_.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end=time.time()
        print((end-start)/size)


    def test_type(self):
        '''
        test by auc value
        :return: auc
        '''
        #self.G.eval()
        y_true, y_pred = self.predict(self.dataloader["test"], scale= False)
        rocprc, rocauc, best_th, best_f1 = evaluate(y_true, y_pred)
        return rocprc, rocauc, best_th, best_f1




