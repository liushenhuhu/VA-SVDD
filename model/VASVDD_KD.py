# 无伪数据增强多分类 消融实验
# 设置α为0，并去除sigmaloss中伪异常部分
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastdtw import fastdtw

from model.Resnet1d import ResNet1D
from model.metric_my import evaluate

dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

def t_softmax(logits, temperature=1):
    """
    logits: 教师模型的输出 logits
    temperature: 温度参数 T
    """
    return F.softmax(logits / temperature, dim=-1)
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

class classificationHead(nn.Module):
    def __init__(self, ngpu, opt):
        super(classificationHead, self).__init__()
        self.classificationHead_l1 = nn.Linear(64, 16, bias=True)
        self.classificationHead_l2 = nn.Linear(16, 3, bias=True)
    def forward(self, z):
        out = self.classificationHead_l1(z)
        out = self.classificationHead_l2(out)
        out = t_softmax(out)
        return out
class Teacher(nn.Module):
    def __init__(self, opt):
        super(Teacher, self).__init__()
        self.temperature = opt.temperature
        self.resnet = ResNet1D(name='resnet18', head='linear', input_channels=1)


        self.clisificationHead = classificationHead(opt.ngpu, opt)

    def forward(self, z):
        with torch.no_grad():
            out = self.resnet(z)
        out = self.clisificationHead(out)
        out = t_softmax(out, temperature = self.temperature)
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

        self.conv_block_T = Encoder(opt.ngpu, opt).to(device)
        self.clisificationHead_T = classificationHead(opt.ngpu, opt).to(device)

        self.conv_block_F = Encoder(opt.ngpu, opt).to(device)
        self.clisificationHead_F = classificationHead(opt.ngpu, opt).to(device)


        self.teacher_T = Teacher(opt).to(device)
        self.teacher_T.resnet.load_state_dict(torch.load("/home/yangliu/project/vf-fake-promax/resnet_T.pth", weights_only=False)["resnet18"])
        for param in self.teacher_T.parameters():
            param.requires_grad = False  # 冻结参数

        self.teacher_F = Teacher(opt).to(device)
        self.teacher_F.resnet.load_state_dict(torch.load("/home/yangliu/project/vf-fake-promax/resnet_F.pth", weights_only=False)["resnet18"])
        for param in self.teacher_F.parameters():
            param.requires_grad = False  # 冻结参数

        # self.teacher_T.load_state_dict(state_dict_load)
        # self.teacher_F.load_state_dict(state_dict_load)

        # self.output_layer = nn.Linear(self.hidden_size, opt.nz * 2)



    def forward(self, x, x_noisy, x_fm, x_f, x_noisy_f, x_fm_f):
        # begin_time = time.time()

        if torch.isnan(x).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        z_x = self.conv_block_T(x)  # z_x:128,64,1
        z_x_noisy = self.conv_block_T(x_noisy)
        z_x_fm = self.conv_block_T(x_fm)

        z_x_f = self.conv_block_F(x_f)
        z_x_noisy_f = self.conv_block_F(x_noisy_f)
        z_x_fm_f = self.conv_block_F(x_fm_f)

        z1 = torch.squeeze(z_x)  # z1:128,64
        z2 = torch.squeeze(z_x_noisy)
        z3 = torch.squeeze(z_x_fm)

        z4 = torch.squeeze(z_x_f)
        z5 = torch.squeeze(z_x_noisy_f)
        z6 = torch.squeeze(z_x_fm_f)

        # 时域和频域分别过分类头
        score_feature1 = self.clisificationHead_T(z1)  # (batchsize,3)
        score_feature2 = self.clisificationHead_T(z2)
        score_feature3 = self.clisificationHead_T(z3)

        score_feature4 = self.clisificationHead_F(z4)
        score_feature5 = self.clisificationHead_F(z5)
        score_feature6 = self.clisificationHead_F(z6)


        teacherT_feature1 = self.teacher_T(x)    #(barchsize,3)
        teacherT_feature2 = self.teacher_T(x_noisy)
        teacherT_feature3 = self.teacher_T(x_fm)


        teacherF_feature1 = self.teacher_F(x_f)  #(barchsize,3)
        teacherF_feature2 = self.teacher_F(x_noisy_f)
        teacherF_feature3 = self.teacher_F(x_fm_f)

        classify_feature = [score_feature1, score_feature2, score_feature3, score_feature4, score_feature5, score_feature6]
        teacher_feature = [teacherT_feature1, teacherT_feature2, teacherT_feature3, teacherF_feature1, teacherF_feature2, teacherF_feature3]

        # print("前向传播时间：",time.time()-begin_time)
        return z1, z2, z3, z4, z5, z6, classify_feature, teacher_feature


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

        self.all_loss_T = []
        self.all_loss_F = []

        # 每epoch loss
        self.all_loss_epoch_T = []
        self.rec_loss_epoch_T = []
        self.pre_loss_epoch_T = []

        self.all_loss_epoch_F = []
        self.rec_loss_epoch_F = []
        self.pre_loss_epoch_F = []

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
        self.x = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        # 输入伪异常
        self.x_noisy = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                   device=self.device)
        self.x_fm = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                device=self.device)
        # 输入频域信号
        self.x_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)
        # 输入频域伪异常
        self.x_noisy_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)
        self.x_fm_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
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
        # self.train_hist = {}
        # self.train_hist['loss_T'] = []
        # self.train_hist['loss_F'] = []
        # self.train_hist['per_epoch_time'] = []
        # self.train_hist['total_time'] = []

        print("Train SVDD_fake_TF")

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

                # self.all_loss_T.append(np.sum(self.loss_t) / len(self.loss_t))
                # self.all_loss_F.append(np.sum(self.loss_f) / len(self.loss_f))

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
                print("SVDD_fake_TF: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre_test, Recall_test,
                                                                                            f1_test))

                # self.scheduler.step()
        # self.train_hist['total_time'].append(time.time() - start_time)

        return ap_test, auc_test, best_auc_epoch, Pre_test, Recall_test, f1_test

    def train_epoch(self):

        self.Backbone.train()
        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()


    def set_input(self, input, istrain=True):
        with torch.no_grad():
            self.x.resize_(input[0].size()).copy_(input[0])
            self.x_noisy.resize_(input[1].size()).copy_(input[1])
            self.x_fm.resize_(input[2].size()).copy_(input[2])
            self.x_f.resize_(input[3].size()).copy_(input[3])
            self.x_noisy_f.resize_(input[4].size()).copy_(input[4])
            self.x_fm_f.resize_(input[5].size()).copy_(input[5])
            self.label.resize_(input[6].size()).copy_(input[6])
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

        # 1.原数据 2.噪声 3.VA 4.频域原数据 5.频域噪声 6.频域VA
        feature1, feature2, feature3, feature4, feature5, feature6, classify_feature,teacher_feature = self.Backbone(
            self.x, self.x_noisy,
            self.x_fm, self.x_f,
            self.x_noisy_f, self.x_fm_f)
        # 1 svdd-loss
        loss_svdd, score = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c1)

        # 2 获取分类头和resnet的 softloss
        loss_t_1 = self.get_softloss(classify_feature[0],teacher_feature[0])
        loss_t_2 = self.get_softloss(classify_feature[1],teacher_feature[1])
        loss_t_3 = self.get_softloss(classify_feature[2],teacher_feature[2])
        loss_t = (loss_t_1 + loss_t_2 + loss_t_3)/3
        loss_f_1 = self.get_softloss(classify_feature[3],teacher_feature[3])
        loss_f_2 = self.get_softloss(classify_feature[4],teacher_feature[4])
        loss_f_3 = self.get_softloss(classify_feature[5],teacher_feature[5])
        loss_f = (loss_f_1 + loss_f_2 + loss_f_3) / 3
        softloss = loss_t*self.opt.tf_percent + loss_f*(1-self.opt.tf_percent)


        # 3 获取resnet hardloss
        # resnet_t_loss1 = self.get_hardloss(teacher_feature[0],self.label)
        # resnet_t_loss2 = self.get_hardloss(teacher_feature[1],self.label_noise)
        # resnet_t_loss3 = self.get_hardloss(teacher_feature[2],self.label_vf)
        # resnet_f_loss1 = self.get_hardloss(teacher_feature[3],self.label)
        # resnet_f_loss2 = self.get_hardloss(teacher_feature[4],self.label_noise)
        # resnet_f_loss3 = self.get_hardloss(teacher_feature[5],self.label_vf)
        # hardloss = (resnet_t_loss1 + resnet_t_loss2 + resnet_t_loss3 + resnet_f_loss1 + resnet_f_loss2 + resnet_f_loss3) / 6


        self.loss = (1 - self.alpha) * loss_svdd + self.alpha * (softloss)

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
                x, x_noisy, x_fm, x_f, x_noisy_f, x_fm_f,_,_,_ = data
                x = x.float().to(self.device)
                x_noisy = x_noisy.float().to(self.device)
                x_fm = x_fm.float().to(self.device)
                x_f = x_f.float().to(self.device)
                x_noisy_f = x_noisy_f.float().to(self.device)
                x_fm_f = x_fm_f.float().to(self.device)

                outputs, z2, z3, outputs_f, z5, z6, _, _ = self.Backbone(x, x_noisy, x_fm, x_f, x_noisy_f, x_fm_f)
                n_samples += outputs.shape[0]
                all_feature = torch.cat((outputs, outputs_f), dim=0)
                # all_feature = outputs
                c += torch.sum(all_feature, dim=0)

        c /= (2 * n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_loss_score(self, feature1, feature2, feature3, feature4, feature5, feature6, center):

        center = center.unsqueeze(0)  # 添加维度
        center = F.normalize(center, dim=1)  # 归一化
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        feature3 = F.normalize(feature3, dim=1)
        feature4 = F.normalize(feature4, dim=1)
        feature5 = F.normalize(feature5, dim=1)
        feature6 = F.normalize(feature6, dim=1)
        feature7 = feature1 + feature5 # 原数据+ 频域噪声
        feature8 = feature4 + feature3 # VA + 频域原数据
        feature9 = feature3 + feature6 # 噪声 + 频域VA

        distance1 = 1 - F.cosine_similarity(feature1, center, eps=1e-6)
        distance2 = 1 - F.cosine_similarity(feature2, center, eps=1e-6)
        distance3 = 1 - F.cosine_similarity(feature3, center, eps=1e-6)
        distance4 = 1 - F.cosine_similarity(feature4, center, eps=1e-6)
        distance5 = 1 - F.cosine_similarity(feature5, center, eps=1e-6)
        distance6 = 1 - F.cosine_similarity(feature6, center, eps=1e-6)
        distance7 = 1 - F.cosine_similarity(feature7, center, eps=1e-6)
        distance8 = 1 - F.cosine_similarity(feature8, center, eps=1e-6)
        distance9 = 1 - F.cosine_similarity(feature9, center, eps=1e-6)

        # 防止模型过拟合
        sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
        sigma_aug2 = torch.sqrt(feature2.var([0]) + 0.0001)
        sigma_aug3 = torch.sqrt(feature3.var([0]) + 0.0001)
        sigma_aug4 = torch.sqrt(feature4.var([0]) + 0.0001)
        sigma_aug5 = torch.sqrt(feature5.var([0]) + 0.0001)
        sigma_aug6 = torch.sqrt(feature6.var([0]) + 0.0001)
        sigma_aug7 = torch.sqrt(feature7.var([0]) + 0.0001)
        sigma_aug8 = torch.sqrt(feature8.var([0]) + 0.0001)
        sigma_aug9 = torch.sqrt(feature9.var([0]) + 0.0001)

        sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
        sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
        sigma_loss3 = torch.max(torch.zeros_like(sigma_aug3), (1 - sigma_aug3))
        loss_sigam1 = torch.mean((sigma_loss1 + sigma_loss2 + sigma_loss3) / 3)

        sigma_loss4 = torch.max(torch.zeros_like(sigma_aug4), (1 - sigma_aug4))
        sigma_loss5 = torch.max(torch.zeros_like(sigma_aug5), (1 - sigma_aug5))
        sigma_loss6 = torch.max(torch.zeros_like(sigma_aug6), (1 - sigma_aug6))
        loss_sigam2 = torch.mean((sigma_loss4 + sigma_loss5 + sigma_loss6) / 3)

        sigma_loss7 = torch.max(torch.zeros_like(sigma_aug7), (1 - sigma_aug7))
        sigma_loss8 = torch.max(torch.zeros_like(sigma_aug8), (1 - sigma_aug8))
        sigma_loss9 = torch.max(torch.zeros_like(sigma_aug9), (1 - sigma_aug9))
        loss_sigam3 = torch.mean((sigma_loss7 + sigma_loss8 + sigma_loss9) / 3)

        # loss_sigam = (loss_sigam1 + loss_sigam2 + loss_sigam3) / 3

        loss_sigam = torch.mean((sigma_loss1 + sigma_loss4)/2)


        a = self.opt.tf_percent #时域占比
        # score
        score = a * distance1 + (1-a)*distance4

        # 总损失
        loss = F.relu(score)
        loss = torch.mean(loss)
        # print("loss:", loss, torch.mean(loss_sigam))
        loss = 1 * loss + self.sigm * torch.mean(loss_sigam)




        #sc方法测试
        # triplet_loss = F.relu(distance1 - distance4 + 0.5) + F.relu(distance1 - distance2 + 0.5)
        # triplet_loss = torch.mean(triplet_loss)
        # triplet_loss2 = F.relu(distance4 - distance3 + 0.5) + F.relu(distance4 - distance6 + 0.5)
        # triplet_loss2 = torch.mean(triplet_loss2)
        # triplet_loss3 = F.relu(distance7 - distance8 + 0.5) + F.relu(distance7 - distance9 + 0.5)
        # triplet_loss3 = torch.mean(triplet_loss3)
        # score = (distance1 + distance4 + distance7) / 3
        # loss = (torch.mean(triplet_loss) + torch.mean(triplet_loss2) + torch.mean(triplet_loss3)) / 3
        # loss = (torch.mean(F.relu(score)) + (torch.mean(triplet_loss) + torch.mean(triplet_loss2) + torch.mean(triplet_loss3)) / 3) / 2
        # loss = 1 * loss + self.sigm * torch.mean(loss_sigam)

        return loss, score

    def get_errors(self):
        errors = {
            # 'loss_pre': self.loss_pre.item(),
            # 'loss_rec': self.loss_rec.item(),
            'loss_f': self.loss_f.item(),
            'loss_t': self.loss_t.item()
        }
        return errors

    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["val"])
        auc_prc, roc_auc, Pre, Recall, f1,_ = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1

    def test(self):
        '''
        vf-fake-promax by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["test"])
        auc_prc, roc_auc, Pre, Recall, f1, _ = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1

    def predict(self, dataloader, scale=False):
        with torch.no_grad():
            # 异常评分
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            # 标签
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)

            for i, data in enumerate(dataloader, 0):
                self.set_input(data, istrain=False)
                #在预测中只有1和4有用
                feature1, feature2, feature3, feature4, feature5, feature6, _, _= self.Backbone(self.x,
                                                                                                    self.x_noisy,
                                                                                                    self.x_fm,
                                                                                                    self.x_f,
                                                                                                    self.x_noisy_f,
                                                                                                    self.x_fm_f)

                _, score = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c1)

                self.an_scores[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score.size(0))] = score.reshape(
                    score.size(0))
                self.gt_labels[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score.size(0))] = self.label.reshape(
                    score.size(0))

            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()

            return y_true, y_pred

    def get_softloss(self, score1, score2):
        # input (batchsize,3)  (batchsize,3)

        loss = torch.sum((score1 - score2) ** 2)

        return loss
    def get_hardloss(self, score, y):
        # # input (batchsize,3)  (batchsize)
        loss = F.cross_entropy(score, y)
        return loss
