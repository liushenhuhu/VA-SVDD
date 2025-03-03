import time,os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .network import AD_MODEL, weights_init
from metric import evaluate
from .plotUtil import save_ts_heatmap_2D, save_ts_heatmap_1D
from TSNE.TSNE import do_tsne, do_tsne_sns
from TSNE.Do_Hist import do_hist


dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))

class Encoder(nn.Module):
    def __init__(self, ngpu, opt, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 4, out_z, 2, 1, 0, bias=False),
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
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh(),

            nn.Linear(80, opt.isize),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.ngpu, opt, 1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

        # self.signal_length = opt.signal_length

    def forward(self, x):    # （64，1，320）
        # features = self.features(x[:,:,:self.signal_length[0]])   # （64，512，10）   (32,512,4)
        features = self.features(x)

        classifier = self.classifier(features)  #  （64，1，1）
        classifier = classifier.view(-1, 1).squeeze(1)   #（64）

        return classifier, features


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        opt.isize

        self.encoder1 = Encoder(opt.ngpu, opt, opt.nz)
        self.decoder = Decoder(opt.ngpu, opt)
        # self.signal_length = opt.signal_length

    def forward(self, x): # （64，1，320）
        # latent_i = self.encoder1(x[:, :, :self.signal_length[0]])  #（64，50，1）
        latent_i = self.encoder1(x)  #（64，50，1）

        gen_x = self.decoder(latent_i)  #（64，1，320）
        return gen_x, latent_i


class BeatGAN(AD_MODEL):

    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.dataset = opt.dataset
        self.model = opt.model
        self.outf = opt.outf
        self.normal_idx = opt.normal_idx
        self.seed = opt.seed

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator( opt).to(device)
        self.G.apply(weights_init)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch=0

        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

        self.tsne_pred = None
        self.tsne_true = None
        self.latent = None


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train BeatGAN.")
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
        Pre_test = 0
        Rec_test = 0

        # with open(os.path.join(self.outf, self.model, self.dataloader, "val_info.txt"), "w") as f:
        #     for epoch in range(self.niter):
        #         self.cur_epoch+=1
        #
        #         # Train
        #         self.train_epoch()
        #
        #
        #             # Val
        #         ap, auc, th, pre, rec, f1 = self.validate()
        #         if (pre + rec) > best_result:
        #
        #             best_result = pre + rec
        #             best_pre = pre
        #             best_rec = rec
        #             best_f1 = f1
        #
        #             best_result_epoch = self.cur_epoch
        #
        #             # Test
        #             ap_test, auc_test, th_test, Pre_test, Rec_test, F1_test = self.test()
        #
        #
        #             #     if epoch == 1:
        #             #         early_stop_auc = auc_test
        #             #
        #             # if auc_test <= early_stop_auc :
        #             #     early_stop_epoch = early_stop_epoch+1
        #             # else:
        #             #     early_stop_epoch = 0
        #             #     early_stop_auc = auc_test
        #             #
        #             # if early_stop_epoch == self.opt.early_stop:
        #             #
        #             #         break
        #
        #             if epoch == 1:
        #                 early_stop_results = (Pre_test + Rec_test)
        #
        #         if (Pre_test + Rec_test ) <= early_stop_results:
        #             early_stop_epoch = early_stop_epoch + 1
        #         else:
        #             early_stop_epoch = 0
        #             early_stop_results = (Pre_test + Rec_test)
        #             if auc_test > 0.9:
        #                 do_tsne(self.latent, self.tsne_true, self.model, self.dataloader, False, self.normal_idx,
        #                         self.seed)
        #
        #         if early_stop_epoch == self.opt.early_stop:
        #             break
        #
        #         f.write(
        #             "EPOCH [{}] Pre:{:.4f} \t  Rec:{:.4f} \t  F1:{:.4f} \t BEST VAL pre:{:.4f} \t  VAL_rec:{:.4f}\t  VAL_f1:{:.4f}  \t in epoch[{}] \t TEST  Pre:{:.4f} \t Rec:{:.4f}\t  F1:{:.4f}\t EarlyStop [{}] \t".format(
        #                 self.cur_epoch, pre, rec, f1, best_pre, best_rec, best_f1, best_result_epoch, Pre_test,
        #                 Rec_test, F1_test, early_stop_epoch))
        #         print(
        #             "EPOCH [{}]  loss:{:.4f} \t Pre:{:.4f} \t  Rec:{:.4f} \t  F1:{:.4f} \t BEST VAL pre:{:.4f} \t  VAL_rec:{:.4f}\t  VAL_f1:{:.4f}  \t in epoch[{}] \t TEST  Pre:{:.4f} \t Rec:{:.4f}\t  F1:{:.4f}\t EarlyStop [{}] \t".format(
        #                 self.cur_epoch, self.err_g, pre, rec, f1, best_pre, best_rec, best_f1, best_result_epoch,
        #                 Pre_test, Rec_test, F1_test, early_stop_epoch))
        #
        #     self.train_hist['total_time'].append(time.time() - start_time)
        #     print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
        #                                                                     self.niter,
        #                                                                     self.train_hist['total_time'][0]))
        #
        # #save_ts_heatmap_2D(self.input.cpu().numpy(),self.fake.cpu().numpy(),'BeatGAN.svg')
        #
        #
        # return ap_test,auc_test,Pre_test,Rec_test, best_result_epoch
        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):

                self.cur_epoch+=1

                # Train
                self.train_epoch()


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

                    #do_tsne_sns(self.tsne_latent, self.tsne_true, self.model, self.dataloader, False, self.normal_idx,self.seed)


                if early_stop_epoch == self.opt.early_stop:
                    break


                f.write("EPOCH [{}] auc:{:.4f} \t  ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                    self.cur_epoch, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test ,early_stop_epoch))
                print( "EPOCH [{}]   \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch,  auc, ap, best_auc, best_ap, best_auc_epoch, auc_test ,ap_test, early_stop_epoch))


        return ap_test, auc_test, best_auc_epoch



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()

            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (epoch_iter  % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
                # print("err_adv:{}  ,err_rec:{}  ,err_enc:{}".format(errors["err_g_adv"],errors["err_g_rec"],errors["err_g_enc"]))


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        with torch.no_grad():
            real_input,fake_output = self.get_generated_x()

            self.visualize_pair_results(self.cur_epoch,
                                        real_input,
                                        fake_output,
                                        is_train=True)

    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])

            # fixed input for view
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    def optimize(self):

        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()

    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)   # (64)
        self.out_d_real, self.feat_real = self.D(self.input)   # (64)   (64,512,10)  #D对原始数据的判别
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)   # (64)
        self.fake, self.latent_i = self.G(self.input)     # (64,1,320)   (64,50,1)  #G的生成结果
        self.out_d_fake, self.feat_fake = self.D(self.fake)   # (64)   (64,512,10)  #D对G的生成的判别结果
        # --

        #定义损失函数
        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device).to(torch.float32))
        # D对原始数据的判别  与 标签 1 之间的 loss   只用了判别结果
        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device).to(torch.float32))
        # D对G的生成的判别  与 标签 0 之间的loss


        self.err_d=self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()  #使用优化算法

    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')

    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input)   #G的生成结果
        self.out_g, self.feat_fake = self.D(self.fake)  #D对G的生成的判别  特征
        _, self.feat_real = self.D(self.input) #D对原始数据的判别 特征


        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv=self.mse_criterion(self.feat_fake,self.feat_real)  # loss for feature matching

        #D对G的生成的判别特征  与  D对原始数据的判别特征  的 loss
        # self.err_g_rec = self.mse_criterion(self.fake, self.input[:,:,:self.opt.signal_length[0]])  # constrain x' to look like x
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x

        self.err_g =  self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()

    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
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
        do_hist(y_pred, y_true, self.model, self.dataset, False, self.normal_idx, self.seed)

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
        with torch.no_grad():
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)

                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            # y_true = self.gt_labels.cpu().numpy()
            # y_pred = self.an_scores.cpu().numpy()

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            self.tsne_latent = self.latent_i.cpu().numpy()
            self.tsne_true = y_true
            self.tsne_pred = y_pred


            return y_true, y_pred

    def predict_for_right(self,dataloader_,min_score,max_score,threshold,save_dir):
        '''

        :param dataloader:
        :param min_score:
        :param max_score:
        :param threshold:
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair=[]
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)

                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))


                # # Save test images.

                batch_input = self.input.cpu().numpy()
                batch_output = self.fake.cpu().numpy()
                ano_score=error.cpu().numpy()
                assert batch_output.shape[0]==batch_input.shape[0]==ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair)>=100:
                        break
                    normal_score=(ano_score[idx]-min_score)/(max_score-min_score)

                    if normal_score>=threshold:
                        test_pair.append((batch_input[idx],batch_output[idx]))

            # print(len(test_pair))
            self.saveTestPair(test_pair,save_dir)

    def test_type(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test"],scale=False)

        over_all=np.concatenate([y_pred_N])
        over_all_gt=np.concatenate([y_N])
        min_score,max_score=np.min(over_all),np.max(over_all)
        A_res={
            "N": y_pred_N
        }
        self.analysisRes(y_pred_N, A_res, min_score, max_score, res_th, save_dir)

        aucprc, aucroc, best_th, best_f1 = evaluate(over_all_gt, (over_all - min_score) / (
                    max_score - min_score))

        print("#############################")
        print("########  Result  ###########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th, best_f1))

        with open(os.path.join(save_dir, "res-record.txt"), 'w') as f:
            f.write("auc_prc:{}\n".format(aucprc))
            f.write("auc_roc:{}\n".format(aucroc))
            f.write("best th:{} --> best f1:{}".format(best_th, best_f1))

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
