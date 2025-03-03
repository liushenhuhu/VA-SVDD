""" Network architectures.
"""

import random
import neurokit2 as nk
# pylint: disable=W0221,W0622,C0103,R0913
import numpy as np
##
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        #assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv1d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm1d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv1d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm1d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv1d(cndf, nz, 4, 1, 0, bias=False))

        main.add_module('Pooling',nn.AdaptiveAvgPool1d(1))

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self,opt,isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        #assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2


        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose1d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm1d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose1d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm1d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv1d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm1d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose1d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())

        main.add_module('Linear',nn.Linear(32,opt.isize))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.decoder_isize, 1, opt.nc, opt.ngf, opt.ngpu)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):#[32,3,32,32]
        features = self.features(x)#[32,256,4,4]  [32,1,87]
        features = features
        classifier = self.classifier(features)#[32,1,1,1] [32,1,1]
        classifier = classifier.view(-1, 1).squeeze(1) #[32]  [32]

        return classifier, features



##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.decoder_isize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
        self.decoder = Decoder(opt,opt.decoder_isize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
        self.encoder2 = Encoder(opt.decoder_isize, opt.nz, opt.nc, opt.ngf, opt.ngpu)





    def mask_waves(self,ecg, maskwhere):
            print('masking...')
            for batch in range(ecg.shape[0]):
                for lead in range(ecg.shape[1]):
                    try:
                        rpeaks = (nk.ecg_peaks(np.array(ecg.to('cpu'))[batch][lead], sampling_rate=100)[1]['ECG_R_Peaks'])
                    except Exception as e:
                        rpeaks = []
                        print(e)
                    mask = ['Q', 'ST', 'T']
                    mask_ecg = ecg
                    if maskwhere == 'Q':
                        for rpeak in rpeaks:
                            if rpeak - 20 >= 0:
                                mask_ecg[batch][lead][rpeak - 20:rpeak] = torch.tensor([(ecg[batch][lead][rpeak - 20] + ecg[batch][lead][rpeak]) / 2] * 20,device=mask_ecg.device)
                            else:
                                mask_ecg[batch][lead][0:rpeak] = torch.tensor([(ecg[batch][lead][0] + ecg[batch][lead][rpeak]) / 2] * rpeak,device=mask_ecg.device)
                    elif maskwhere == 'ST':
                        for rpeak in rpeaks:
                            if rpeak + 20 < ecg.shape[2]:
                                mask_ecg[batch][lead][rpeak:rpeak + 20] = torch.tensor([(ecg[batch][lead][rpeak] + ecg[batch][lead][rpeak + 20]) / 2] * 20,device=mask_ecg.device)
                            else:
                                mask_ecg[batch][lead][rpeak:ecg.shape[2]] = torch.tensor([(ecg[batch][lead][-1] + ecg[batch][lead][rpeak]) / 2] * (ecg.shape[2] - rpeak),device=mask_ecg.device)
                    elif maskwhere == 'T':
                        for rpeak in rpeaks:
                            if rpeak + 20 < ecg.shape[2] & rpeak + 35 < ecg.shape[2]:
                                mask_ecg[batch][lead][rpeak + 20:rpeak + 35] = torch.tensor([(ecg[batch][lead][rpeak + 35] + ecg[batch][lead][rpeak + 20]) / 2] * 15,device=mask_ecg.device)
                            elif rpeak + 20 < ecg.shape[2] & rpeak + 35 >= ecg.shape[2]:
                                mask_ecg[batch][lead][rpeak + 20:ecg.shape[2]] = torch.tensor([(ecg[batch][lead][-1] + ecg[batch][lead][rpeak + 20]) / 2] * (
                                            ecg.shape[2] - rpeak - 20),device=mask_ecg.device)
                            else:
                                print('无可mask的T波')
                    elif maskwhere == 'random':
                        for rpeak in rpeaks:
                            mask_ = random.choice(mask)
                            if mask_ == 'Q':
                                if rpeak - 20 >= 0:
                                    mask_ecg[batch][lead][rpeak - 20:rpeak] = torch.tensor([(ecg[batch][lead][rpeak - 20] + ecg[batch][lead][rpeak]) / 2] * 20,
                                                                              device=mask_ecg.device)
                                else:
                                    mask_ecg[batch][lead][0:rpeak] = torch.tensor([(ecg[batch][lead][0] + ecg[batch][lead][rpeak]) / 2] * rpeak,
                                                                     device=mask_ecg.device)
                            elif mask_ == 'ST':
                                if rpeak + 20 < ecg.shape[2]:
                                    mask_ecg[batch][lead][rpeak:rpeak + 20] = torch.tensor([(ecg[batch][lead][rpeak] + ecg[batch][lead][rpeak + 20]) / 2] * 20,
                                                                              device=mask_ecg.device)
                                else:
                                    mask_ecg[batch][lead][rpeak:ecg.shape[2]] = torch.tensor(
                                        [(ecg[batch][lead][-1] + ecg[batch][lead][rpeak]) / 2] * (ecg.shape[2] - rpeak), device=mask_ecg.device)
                            else:
                                if rpeak + 20 < ecg.shape[2] & rpeak + 35 < ecg.shape[2]:
                                    mask_ecg[batch][lead][rpeak + 20:rpeak + 35] = torch.tensor(
                                        [(ecg[batch][lead][rpeak + 35] + ecg[batch][lead][rpeak + 20]) / 2] * 15, device=mask_ecg.device)
                                elif rpeak + 20 < ecg.shape[2] & rpeak + 35 >= ecg.shape[2]:
                                    mask_ecg[batch][lead][rpeak + 20:ecg.shape[2]] = torch.tensor([(ecg[batch][lead][-1] + ecg[batch][lead][rpeak + 20]) / 2] * (
                                            ecg.shape[2] - rpeak - 20), device=mask_ecg.device)
                                else:
                                    print('无可mask的T波')







            return ecg,mask_ecg




    def forward_encoder(self, x1,x2):


        latent_i = self.encoder1(x1)
        latent_j = self.encoder1(x2)

        return latent_i,latent_j

    def forward_encoder2(self, x1, x2):

        latent_i = self.encoder2(x1)
        latent_j = self.encoder2(x2)

        return latent_i, latent_j

    def forward_decoder(self, latent1,latent2):

        return self.decoder(latent1),self.decoder(latent2)


    def forward_loss(self, real, pred, mask):

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, ecg,real_ecg):
        latent1,latent2 = self.forward_encoder(ecg,real_ecg)
        gen_ecg1,gen_ecg2 = self.forward_decoder(latent1,latent2)  # [N, L, p*p*3]
        latent3,latent4 = self.forward_encoder2(gen_ecg1,gen_ecg2)  # [32,100,1,1]
        #loss = self.forward_loss(ecg, gen_ecg, mask)
        #return gen_ecg,latent,latent_o,0
        return latent1,latent2,latent3,latent4,gen_ecg1,gen_ecg2,0

        #return gen_ecg1

    # def forward(self, x): #[32,3,32,32]  [32,1,720]
    #     latent_i = self.encoder1(x)  #[32,100,1,1]  [32,100,1]
    #     gen_imag = self.decoder(latent_i) #[32,3,32,32]
    #     latent_o = self.encoder2(gen_imag) #[32,100,1,1]
    #     return gen_imag, latent_i, latent_o
