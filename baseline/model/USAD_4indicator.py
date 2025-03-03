import torch
import torch.nn as nn
from baseline.dataloader.UCR_dataloader import load_data
from options import Options
import numpy as np
from baseline.model.metric import evaluate
from baseline.utils_plot.draw_utils import plot_hist,plot_tsne_sns
#from utils import *
device = torch.device("cuda:1" if
torch.cuda.is_available() else "cpu")

opt = Options().parse()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


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
            nn.Linear(80, opt.isize),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out


class UsadModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt.ngpu, opt, opt.nz)
        self.decoder1 = Decoder(opt.ngpu, opt)
        self.decoder2 = Decoder(opt.ngpu, opt)

    def training_step(self, batch, n):
        z = self.encoder(batch)  #(64,1,8)
        w1 = self.decoder1(z)  #(64,1,1014)
        w2 = self.decoder2(z)  #(64,1,1024)
        w3 = self.decoder2(self.encoder(w1))  #(64,1,1024)
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(data[0], device), n) for data in val_loader]
    ap_val, auc_val, pre, rec, f1 = validating(model, val_loader, .5, .5)
    return model.validation_epoch_end(outputs), ap_val, auc_val, pre, rec, f1


def training(opt, model, train_loader, val_loader, test_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))

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
    best_auc_epoch = 0

    for epoch in range(opt.niter):
        for data in train_loader:
            batch = to_device(data[0], device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result, ap_val, auc_val, pre, rec, f1 = evaluate(model, val_loader, epoch + 1)
        #print('AP of val is {}, AUC of val is {}'.format(ap_val, auc_val))
        model.epoch_end(epoch, result)



        if (ap_val+auc_val) > best_result:

            best_result = ap_val+auc_val
            best_pre = pre
            best_rec = rec
            best_f1 = f1
            best_ap=ap_val
            best_auc=auc_val

            best_result_epoch = epoch

            ap_test, auc_test, Pre_test, Rec_test, F1_test = testing(model,opt, test_loader, .5, .5)
            torch.save(model,
                       '/home/changhuihui/learn_project/COCA/baseline/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed))


            if epoch == 1:
                early_stop_results = ap_test+auc_test

        if ap_test+auc_test <= early_stop_results:
            early_stop_epoch = early_stop_epoch + 1
        else:
            early_stop_epoch = 0
            early_stop_results = ap_test+auc_test
            # torch.save(model,
            #        './Model_CheckPoint/{}_{}_{}_{}.pkl'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed))

        if early_stop_epoch == opt.early_stop:
            break

        print( "EPOCH [{}] AUC:{:.4f} \t  AP:{:.4f} \t BEST VAL AUC:{:.4f} \t  VAL_AP:{:.4f}\t in epoch[{}] \t TEST  AUC:{:.4f} \t AP:{:.4f}\t EarlyStop [{}] \t".format(
                epoch, auc_val, ap_val, best_auc, best_ap,best_result_epoch, auc_test, ap_test , early_stop_epoch))

        # print( "EPOCH [{}]   \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
        #         epoch, auc_val, ap_val, best_auc, best_ap, best_auc_epoch, auc_test, ap_test, early_stop_epoch))

        #print('AP of test is {}, AUC of test is {}'.format(ap_test, auc_test))


        #history.append(result)

    return ap_test, auc_test,Pre_test,Rec_test, best_result_epoch

    #return ap_test, auc_test, best_auc_epoch


def testing(model,opt, test_loader, alpha=.5, beta=.5):
    results = []
    labels = []
    latent1_all = []
    latent2_all = []
    for data in test_loader:
        batch = to_device(data[0], device)
        latent1=model.encoder(batch)
        w1 = model.decoder1(latent1)
        latent2=model.encoder(w1)
        w2 = model.decoder2(model.encoder(w1))
 
        batch = torch.squeeze(batch)
        w1,w2 = torch.squeeze(w1), torch.squeeze(w2)

        batch = batch.view(batch.shape[0], -1)
        w1 = w1.view(w1.shape[0], -1)
        w2 = w2.view(w2.shape[0], -1)

        result = alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1)
        result = torch.flatten(result).cpu().detach().numpy()
        results.extend(result)
        latent1_all.extend(torch.squeeze(latent1).cpu().detach().numpy())
        latent2_all.extend(torch.squeeze(latent2).cpu().detach().numpy())
        label = data[1].cpu().detach().numpy()
        labels.extend(label)
    results=np.array(results)
    labels = np.array(labels)
    latent1_all = np.array(latent1_all)
    latent2_all = np.array(latent2_all)
    if opt.istest == True:
        plot_hist(results, labels, opt, 'USAD_4indicator_hist',display=False)
        plot_tsne_sns(latent1_all, labels.astype(int), opt,'USAD_4indicator_latent1', display=False)
        plot_tsne_sns(latent2_all, labels.astype(int), opt,'USAD_4indicator_latent2', display=False)

    results = np.array(results)
    labels = np.array(labels)

    ap, auc, pre, rec, f1 = evaluation(labels,results)

    return ap, auc,pre, rec, f1



def validating(model, val_loader, alpha=.5, beta=.5):
    results = []
    labels = []
    for data in val_loader:
        batch = to_device(data[0], device)

        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))

        batch = torch.squeeze(batch)
        w1,w2 = torch.squeeze(w1), torch.squeeze(w2)

        batch = batch.view(batch.shape[0],-1)
        w1 = w1.view(w1.shape[0],-1)
        w2 = w2.view(w2.shape[0],-1)


        result = alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1)
        result = torch.flatten(result).cpu().detach().numpy()
        results.extend(result)


        label = data[1].cpu().detach().numpy()
        labels.extend(label)


    results = np.array(results)
    labels = np.array(labels)

    ap, auc, pre, rec, f1 = evaluation(labels,results)

    return ap, auc,pre, rec, f1




# if __name__ == '__main__':
#
#
#     dataloader, opt.isize, opt.signal_length = load_data(opt, 'CWRU')
#     model = UsadModel(opt)
#     model = model.to(device)
#
#
#     _,_,_ = training(100,model,dataloader['train'], dataloader['val'], dataloader['test'])
#
