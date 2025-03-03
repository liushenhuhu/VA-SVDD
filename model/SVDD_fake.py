# 变动
# 更改SVDD loss 进行实验
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastdtw import fastdtw

from model.metric_my import evaluate

dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))


class Encoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),  # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),  # 32,64
            nn.BatchNorm1d(opt.ndf * 2),  # 归一化处理 参数为需要归一化的维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  # 64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 4, opt.nz, 2, 1, 0, bias=False),  # 128,50
            nn.AdaptiveAvgPool1d(1)  # 平均池化，将最低维度数转化为1个数  128,1
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
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),
            # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, 30),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out


class Encoder1(nn.Module):
    def __init__(self, opt):
        super(Encoder1, self).__init__()
        self.nc = opt.nc
        self.linear = nn.Linear(opt.nc * 30, opt.nz)

    def forward(self, x):
        x = x.view(-1, self.nc * 30)
        x = self.linear(x)
        return x


class Encoder_m(nn.Module):
    def __init__(self, ngpu, opt):
        super(Encoder_m, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(1, opt.ndf, 4, 2, 1, bias=False),  # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),  # 32,64
            nn.BatchNorm1d(opt.ndf * 2),  # 归一化处理 参数为需要归一化的维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  # 64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 4, opt.nz_m, 2, 1, 0, bias=False),  # 128,50
            nn.AdaptiveAvgPool1d(1)  # 平均池化，将最低维度数转化为1个数
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder_m(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder_m, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz_m, opt.ngf * 4, 10, 1, 0, bias=False),
            # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, 1, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, opt.nz * 2),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out
class classificationHead(nn.Module):
    def __init__(self, ngpu, opt):
        super(classificationHead, self).__init__()
        self.classificationHead_l1 = nn.Linear(64, 16, bias=True)
        self.classificationHead_l2 = nn.Linear(16, 3, bias=True)
    def forward(self, z):
        out = self.classificationHead_l1(z)
        out = self.classificationHead_l2(out)
        return out
class base_Model(nn.Module):
    def __init__(self, opt, device):
        super(base_Model, self).__init__()
        self.input_channels = opt.nc
        self.final_out_channels = 16
        self.features_len = 24
        self.project_channels = 20
        self.hidden_size = 64
        self.window_size = 16
        self.device = device
        self.num_layers = 3
        self.kernel_size = 4
        self.stride = 1
        self.dropout = 0.455
        self.nz = opt.nz  # opt.nz =64

        self.conv_block = Encoder(opt.ngpu, opt).to(device)
        self.clisificationHead = classificationHead(opt.ngpu, opt).to(device)
        # self.output_layer = nn.Linear(self.hidden_size, opt.nz * 2)



    def forward(self, x_t, x_f, x_noisy, n_f, x_vf, v_f):
        if torch.isnan(x_t).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        z_x = self.conv_block(x_t)  # z_x:128,64,1
        z_x_f = self.conv_block(x_f)
        z_x_f = z_x_f.view(128, self.nz, 1)
        # z2 = z2.view(128, self.nz, 1)
        z_n = self.conv_block(x_noisy)
        z_n_f = self.conv_block(n_f)
        z_n_f = z_n_f.view(128, self.nz, 1)
        z_x_vf = self.conv_block(x_vf)
        z_v_f = self.conv_block(v_f)
        z_v_f = z_v_f.view(128, self.nz, 1)

        z1 = torch.squeeze(z_x)  # z1:128,64
        z2 = torch.squeeze(z_x_f)
        z3 = torch.squeeze(z_n)
        z4 = torch.squeeze(z_n_f)
        z5 = torch.squeeze(z_x_vf)
        z6 = torch.squeeze(z_v_f)

        score_feature1 = self.clisificationHead(z1)  # 128,3
        score_feature3 = self.clisificationHead(z3)
        score_feature5 = self.clisificationHead(z5)

        return z1, z2, z3, z4, z5, z6, score_feature1, score_feature3, score_feature5


class ModelTrainer(nn.Module):
    def __init__(self, opt, dataloader, device):
        super(ModelTrainer, self).__init__()
        self.alpha = opt.alpha
        self.niter = opt.niter  # 训练次数 #1000
        self.dataset = opt.dataset  # 数据集
        self.model = opt.model
        self.outf = opt.outf  # 输出文件夹路径

        self.dataloader = dataloader
        self.device = device
        self.opt = opt
        self.sigm = opt.sigm

        self.all_loss = []
        # self.rec_loss = []
        # self.pre_loss = []
        # 每epoch loss
        self.all_loss_epoch = []
        self.rec_loss_epoch = []
        self.pre_loss_epoch = []

        self.batchsize = opt.batchsize  # input batch size 32
        self.nz = opt.nz  # 潜在z向量大小 8
        self.niter = opt.niter  #1000

        self.Backbone = base_Model(opt, device).to(device)

        self.bc_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            [
                {"params": self.Backbone.parameters()}],
            lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0

        self.c1 = torch.zeros(self.opt.nz, device=self.device)

        # 输入时序信号
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        # 输入伪异常
        self.x_noisy = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                   device=self.device)
        self.x_vf = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                device=self.device)
        # 输入频域信号
        self.input_hrv = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.signal_length[0]),
                                     dtype=torch.float32, device=self.device)
        # 输入频域伪异常
        self.n_wavelet = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)
        self.v_wavelet = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)
        # 三种标签 0正常 1噪声
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)
        self.label_noise = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)
        self.label_vf = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)

        self.latent_i_raw = None
        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def train(self):
        self.train_hist = {}
        self.train_hist['loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train FT_svdd3_fake")

        start_time = time.time()
        best_result = 0
        best_ap = 0
        best_auc = 0
        best_auc_epoch = 0

        early_stop_epoch = 0
        early_stop_auc = 0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch += 1

                # Train
                self.train_epoch()

                self.all_loss_epoch.append(np.sum(self.all_loss) / len(self.all_loss))
                # self.rec_loss_epoch.append(np.sum(self.rec_loss) / len(self.rec_loss))
                # self.pre_loss_epoch.append(np.sum(self.pre_loss) / len(self.pre_loss))
                # self.all_loss = []
                # self.rec_loss = []
                # self.pre_loss = []

                # Val
                ap, auc, Pre, Recall, f1 = self.validate()

                if auc > best_result:
                    best_result = auc
                    best_auc = auc
                    best_ap = ap
                    best_auc_epoch = self.cur_epoch

                    # Test
                    ap_test, auc_test, Pre_test, Recall_test, f1_test = self.test()

                    if epoch == 1:
                        early_stop_auc = auc_test

                if auc_test <= early_stop_auc:
                    early_stop_epoch = early_stop_epoch + 1
                else:
                    early_stop_epoch = 0
                    early_stop_auc = auc_test

                if early_stop_epoch == self.opt.early_stop:
                    break

                if epoch < 1:
                    self.c1 = self.center_c(self.dataloader["train"])


                f.write(
                    "EPOCH [{}] auc:{:.4f} \t  ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print(
                    "EPOCH [{}]  loss:{:.4f} \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, self.loss, auc, ap, best_auc, best_ap,
                        best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print("val: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre, Recall, f1))
                print("vf-fake-promax: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre_test, Recall_test,
                                                                                            f1_test))
                # self.scheduler.step()
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        # with open(os.path.join(self.outf, self.model, self.dataset, "loss.txt"), "a+") as f:
        #     f.write("\nall_loss\n")
        #     f.write(str(self.all_loss))
        #     f.write("\nrec_loss\n")
        #     f.write(str(self.rec_loss))
        #     f.write("\npre_loss\n")
        #     f.write(str(self.pre_loss))

        return ap_test, auc_test, best_auc_epoch, Pre_test, Recall_test, f1_test

    def train_epoch(self):

        epoch_start_time = time.time()
        self.Backbone.train()

        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()
            errors = self.get_errors()
            self.train_hist['loss'].append(errors["loss"])
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

    def set_input(self, input, istrain=True):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.input_hrv.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[2].size()).copy_(input[2])
            self.x_noisy.resize_(input[3].size()).copy_(input[3])
            self.n_wavelet.resize_(input[4].size()).copy_(input[4])
            self.x_vf.resize_(input[5].size()).copy_(input[5])
            self.v_wavelet.resize_(input[6].size()).copy_(input[6])
            if (istrain):
                self.label_noise.resize_(input[7].size()).copy_(input[7])
                self.label_vf.resize_(input[8].size()).copy_(input[8])

    def optimize(self):
        self.update_net()

    def DTWLoss(self, x, y):
        return fastdtw(x, y)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def update_net(self):
        self.optimizer.zero_grad()

        # 1.原数据 2.小波分解 3.噪声 4.噪声+小波分解 5.FM调制信号 6.FM调制信号+小波分解
        feature1, feature2, feature3, feature4, feature5, feature6, score_nomal, score_noise, score_vf = self.Backbone(
            self.input, self.input_hrv,
            self.x_noisy, self.n_wavelet,
            self.x_vf, self.v_wavelet)

        loss1, score1 = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c1)

        # 增加分类任务，对feature1,3,5分类
        loss2 = self.get_classification_loss_score(score_nomal, score_noise, score_vf, self.label, self.label_noise,
                                                   self.label_vf)

        self.loss = (1 - self.alpha) * loss1 + self.alpha * loss2

        self.all_loss.append(self.loss.item())
        self.loss.backward()
        self.optimizer.step()

    def center_c(self, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(64, device=self.device)

        self.Backbone.eval()

        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                input, input_hrv, label, x_noisy, n_f, x_vf, v_f, _, _ = data
                input = input.float().to(self.device)
                input_hrv = input_hrv.float().to(self.device)
                x_noisy = x_noisy.float().to(self.device)
                n_f = n_f.float().to(self.device)
                x_vf = x_vf.float().to(self.device)
                v_f = v_f.float().to(self.device)

                outputs, dec, z3, z4, z5, z6, _, _, _ = self.Backbone(input, input_hrv, x_noisy, n_f, x_vf, v_f)
                n_samples += outputs.shape[0]
                all_feature = torch.cat((outputs, z3), dim=0)
                # all_feature = outputs
                c += torch.sum(all_feature, dim=0)

        c /= (2 * n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_loss_score(self, feature1, feature2, feature3, feature4, feature5, feature6, center):
        # 1.原数据 2.小波分解 3.噪声 4.噪声+小波分解 5.FM调制信号 6.FM调制信号+小波分解

        center = center.unsqueeze(0)  # 添加维度
        center = F.normalize(center, dim=1)  # 归一化
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        feature3 = F.normalize(feature3, dim=1)
        feature4 = F.normalize(feature4, dim=1)
        feature5 = F.normalize(feature5, dim=1)
        feature6 = F.normalize(feature6, dim=1)
        feature7 = feature1 + feature4
        feature8 = feature2 + feature5
        feature9 = feature3 + feature6

        distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
        distance1 = 1 - distance1


        # 防止模型过拟合
        sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
        sigma_aug2 = torch.sqrt(feature2.var([0]) + 0.0001)
        sigma_aug3 = torch.sqrt(feature3.var([0]) + 0.0001)
        sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
        sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
        sigma_loss3 = torch.max(torch.zeros_like(sigma_aug3), (1 - sigma_aug3))
        loss_sigam1 = torch.mean((sigma_loss1 + sigma_loss2 + sigma_loss3) / 3)

        sigma_aug4 = torch.sqrt(feature4.var([0]) + 0.0001)
        sigma_aug5 = torch.sqrt(feature5.var([0]) + 0.0001)
        sigma_aug6 = torch.sqrt(feature6.var([0]) + 0.0001)
        sigma_loss4 = torch.max(torch.zeros_like(sigma_aug4), (1 - sigma_aug4))
        sigma_loss5 = torch.max(torch.zeros_like(sigma_aug5), (1 - sigma_aug5))
        sigma_loss6 = torch.max(torch.zeros_like(sigma_aug6), (1 - sigma_aug6))
        loss_sigam2 = torch.mean((sigma_loss4 + sigma_loss5 + sigma_loss6) / 3)

        sigma_aug7 = torch.sqrt(feature7.var([0]) + 0.0001)
        sigma_aug8 = torch.sqrt(feature8.var([0]) + 0.0001)
        sigma_aug9 = torch.sqrt(feature9.var([0]) + 0.0001)
        sigma_loss7 = torch.max(torch.zeros_like(sigma_aug7), (1 - sigma_aug7))
        sigma_loss8 = torch.max(torch.zeros_like(sigma_aug8), (1 - sigma_aug8))
        sigma_loss9 = torch.max(torch.zeros_like(sigma_aug9), (1 - sigma_aug9))
        loss_sigam3 = torch.mean((sigma_loss7 + sigma_loss8 + sigma_loss9) / 3)

        loss_sigam = (loss_sigam1 + loss_sigam2 + loss_sigam3) / 3

        # The Loss function that representations reconstruction
        # score = (distance1 + distance4 + distance7) / 3
        score = distance1
        # 总损失

        loss = F.relu(distance1)
        loss = torch.mean(loss)
        loss = 1 * loss + self.sigm * torch.mean(loss_sigam)


        # loss = 1 * distance1 + self.sigm * loss_sigam1
        # loss = 1 * loss_oc + 0.1 * loss_sigam

        return loss, score

    def get_errors(self):
        errors = {
            # 'loss_pre': self.loss_pre.item(),
            # 'loss_rec': self.loss_rec.item(),
            'loss': self.loss.item()
        }
        return errors

    def test(self):
        '''
        vf-fake-promax by auc value
        :return: auc
        '''
        y_true, y_pred, latent = self.predict(self.dataloader["test"])
        auc_prc, roc_auc, Pre, Recall, f1, _ = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1

    def test_draw(self):
        '''
        vf-fake-promax by auc value
        :return: auc
        '''
        y_true, y_pred, latent = self.predict(self.dataloader["test"])
        return y_true, y_pred, latent

    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_true, y_pred, latent = self.predict(self.dataloader["val"])
        auc_prc, roc_auc, Pre, Recall, f1, _ = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1

    def predict(self, dataloader, scale=False):
        with torch.no_grad():
            # 异常评分
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores1 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores2 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores3 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.latent_i = torch.zeros(size=(len(dataloader.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            # 标签
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)

            for i, data in enumerate(dataloader, 0):
                self.set_input(data, istrain=False)
                feature1, feature2, feature3, feature4, feature5, feature6, _, _, _ = self.Backbone(self.input,
                                                                                                    self.input_hrv,
                                                                                                    self.x_noisy,
                                                                                                    self.n_wavelet,
                                                                                                    self.x_vf,
                                                                                                    self.v_wavelet)

                self.latent_i = feature1
                loss1, score1 = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c1)

                self.an_scores1[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = score1.reshape(
                    score1.size(0))
                self.gt_labels[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = self.label.reshape(
                    score1.size(0))


                self.an_scores = self.an_scores1

            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            latent = self.latent_i.cpu().numpy()

            return y_true, y_pred, latent

    def get_classification_loss_score(self, score_nomal, score_noise, score_vf, y, y_noise, y_vf):
        # score 128,3  label 128 直接给出0-2的下标即可

        score = torch.cat((score_nomal, score_noise, score_vf), 0)
        label = torch.cat((y, y_noise, y_vf), 0)
        loss = F.cross_entropy(score, label)

        return loss
