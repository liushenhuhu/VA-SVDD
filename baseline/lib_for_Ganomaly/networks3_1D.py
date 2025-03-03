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

import torch.nn as nn
import torch
from torchsummary import summary


# ----------
#  U-NET
# ----------

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, kernel_size=ksize,
                            stride=stride, bias=False, padding_mode='replicate')]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, ksize=4, stride=2, output_padding=0, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose1d(in_size, out_size, kernel_size=ksize,
                               stride=stride, output_padding=output_padding, bias=False),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


# ----------
#  Generator
# ----------



# --------------
#  Discriminator
# --------------


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

    def __init__(self,in_channels=12):
        super(Encoder, self).__init__()

        self.down1 = UNetDown(in_channels, 128, normalize=False)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512, dropout=0.5)
        self.down4 = UNetDown(512, 512, dropout=0.5, normalize=False)

    def forward(self, input):
        d1=self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        return d1,d2,d3,d4

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self,out_channels=12):
        super(Decoder, self).__init__()
        self.up1 = UNetUp(512, 512, output_padding=1, dropout=0.5)
        self.up2 = UNetUp(1024, 256, output_padding=0)
        self.up3 = UNetUp(512, 128, output_padding=1)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(256, out_channels, 3, padding=1,
                      padding_mode='replicate'),
            nn.Tanh(),
        )

    def forward(self, d1,d2,d3,d4):
        u1 = self.up1(d4,d3)
        u2 = self.up2(u1,d2)
        u3 = self.up3(u2,d1)
        final = self.final(u3)
        return final


##
# class NetD(nn.Module):
#     """
#     DISCRIMINATOR NETWORK
#     """
#
#     def __init__(self, opt):
#         super(NetD, self).__init__()
#         model = Encoder()
#         layers = list(model.main.children())
#
#         self.features = nn.Sequential(*layers[:-1])
#         self.classifier = nn.Sequential(layers[-1])
#         self.classifier.add_module('Sigmoid', nn.Sigmoid())
#
#     def forward(self, x):#[32,3,32,32]
#         features = self.features(x)#[32,256,4,4]  [32,1,87]
#         features = features
#         classifier = self.classifier(features)#[32,1,1,1] [32,1,1]
#         classifier = classifier.view(-1, 1).squeeze(1) #[32]  [32]
#
#         return classifier, features



class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt,in_channels=12):
        super(NetD, self).__init__()

        def discriminator_block(in_filters, out_filters, ksize=6, stride=3, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, ksize,
                                stride=stride, padding_mode='replicate')]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 128, normalization=False),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv1d(512, 1, 4, bias=False, padding_mode='replicate')
        )

    def forward(self, signal_A, signal_B):
        # Concatenate signals and condition signals by channels to produce input
        signal_input = torch.cat((signal_A, signal_B), 1)
        return self.model(signal_input)


##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder()
        self.decoder = Decoder()
        self.encoder2 = Encoder()





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


        d11,d12,d13,d14 = self.encoder1(x1)
        d21,d22,d23,d24 = self.encoder1(x2)

        return d11,d12,d13,d14,d21,d22,d23,d24

    def forward_encoder2(self, x1, x2):

        d11, d12, d13, d14 = self.encoder1(x1)
        d21, d22, d23, d24 = self.encoder1(x2)

        return d14,d24

    def forward_decoder(self, d11,d12,d13,d14,d21,d22,d23,d24):

        return self.decoder(d11,d12,d13,d14),self.decoder(d21,d22,d23,d24)


    def forward_loss(self, real, pred, mask):

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, ecg,real_ecg):
        d11,d12,d13,d14,d21,d22,d23,d24 = self.forward_encoder(ecg,real_ecg)
        gen_ecg1,gen_ecg2 = self.forward_decoder(d11,d12,d13,d14,d21,d22,d23,d24)  # [N, L, p*p*3]
        latent3,latent4 = self.forward_encoder2(gen_ecg1,gen_ecg2)  # [32,100,1,1]
        #loss = self.forward_loss(ecg, gen_ecg, mask)
        #return gen_ecg,latent,latent_o,0
        return d14,d24,latent3,latent4,gen_ecg1,gen_ecg2,0

    # def forward(self, x): #[32,3,32,32]  [32,1,720]
    #     latent_i = self.encoder1(x)  #[32,100,1,1]  [32,100,1]
    #     gen_imag = self.decoder(latent_i) #[32,3,32,32]
    #     latent_o = self.encoder2(gen_imag) #[32,100,1,1]
    #     return gen_imag, latent_i, latent_o
