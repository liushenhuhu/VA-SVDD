import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

BASEDIR = os.path.dirname(os.path.abspath(__file__))
from sklearn import metrics
from model.Resnet1d import ResNet1D
import os
import random

from ecg_dataset.resnet_dataloader import get_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
from options import Options
import numpy as np
import datetime

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

#
DATASETS_NAME = {
    'vfdb': 1,
    'mitbih': 1,
    'cudb': 1
}

SEEDS = [
    # 0, 1, 2
    # 0, 2, 3
    1, 2, 3, 4, 5, 6,
    # 1
]
opt.numclass = DATASETS_NAME['vfdb']
opt.dataset = 'vfdb'
opt.seed = 1
opt.is_all_data = True
opt.model = 'VASVDD_KD'
opt.batchsize = 64

# 参数设置
random_seed = 3
BATCH_SIZE = 64
LR = 0.1
lr_decay_step = 10

MAX_EPOCH = 100
log_interval = 30
val_interval = 1
classes = 2
start_epoch = -1

# # 构建DataLoder
dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

# 模型

# 1/3 构建模型

resnet18_ft = ResNet1D(name='resnet18', head='linear', input_channels=1)
# resnet18_ft.load_state_dict(torch.load("/home/yangliu/project/vf-fake-promax/resnet.pth", weights_only=False)["resnet18"])
class classificationHead(nn.Module):
    def __init__(self):
        super(classificationHead, self).__init__()
        self.classificationHead_l1 = nn.Linear(64, 16, bias=True)
        self.classificationHead_l2 = nn.Linear(16, 3, bias=True)
    def forward(self, z):
        out = self.classificationHead_l1(z)
        out = self.classificationHead_l2(out)
        return out
classifier = classificationHead()
classifier = classifier.to(device)
resnet18_ft = resnet18_ft.to(device)
# 2/3 加载参数
flag = 0
if flag:
    path_pretrained_model = os.path.join(BASEDIR, "../dataset/data/resnet18-5c106cde.pth")
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)  # 将参数加载到模型中

# 损失函数
criterion = nn.CrossEntropyLoss()  # 选择损失函数


# 选择优化器
# optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)
optimizer = torch.optim.Adam(resnet18_ft.parameters(), lr=LR, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.5)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(start_epoch + 1, MAX_EPOCH):
    loss_list = []
    correct = 0.
    total = 0.

    resnet18_ft.train()
    for data in dataloader["train"]:
        # set input
        inputs = torch.cat((data[0],data[1],data[2]),0)
        inputs = inputs.to(device)
        labels = torch.cat((data[6],data[7],data[8]),0)
        labels = labels.to(device)
        # randomshuffle?

        outputs = classifier(resnet18_ft(inputs.float())) # (batchsize*6,3)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels.long())
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()
        loss_list.append(loss.item())

    loss_mean = np.mean(loss_list)

    print("Training:Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            epoch, MAX_EPOCH, loss_mean, correct / total))


    scheduler.step()  # 更新学习率

    # #保存模型权重
    # checkpoint = {"model_state_dict": resnet18_ft.state_dict(),
    #               "optimizer_state_dict": optimizer.state_dict(),
    #               "epoch": epoch}
    # PATH = f'./models/checkpoint_{epoch}_epoch.pkl'
    # if not os.path.exists("./models"):
    #     os.makedirs("./models")
    # torch.save(checkpoint, PATH)

    # validate the models
    auc_val = 0.
    correct_val = 0.
    total_val = 0.
    loss_val = 0.
    y_true = []
    y_scores = []
    resnet18_ft.eval()
    maxACC = 0
    with torch.no_grad():

        for data in dataloader["val"]:
            # set input
            inputs = torch.cat((data[0], data[1], data[2]), 0)
            inputs = inputs.to(device)
            labels = torch.cat((data[6], data[7], data[8]), 0)
            labels = labels.to(device)

            outputs = classifier(resnet18_ft(inputs.float()))  # (batchsize*6,3)

            loss = criterion(outputs, labels.long())

            _, predicted = torch.max(outputs.data, 1)  # 获取行最大值下标，即为0、1、2，表示概率更大的一个结果



            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().cpu().sum().numpy()
            loss_val += loss.item()
            labels = list(labels.cpu().numpy())
            y_true.extend(list(map(int, labels)))

            y_scores.extend(outputs.data.cpu().numpy()[:, 0])
        loss_val_mean = loss_val / len(dataloader["val"])
        valid_curve.append(loss_val_mean)

        # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=0)
        # auc_val = metrics.auc(fpr, tpr)
        acc = correct_val / total_val
        if acc>maxACC:
            maxACC = acc
            result = {}
            result["resnet18"] = resnet18_ft.state_dict()
            result["acc"] = acc
            torch.save(result, './resnet_T.pth')
        print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            epoch, MAX_EPOCH, loss_val_mean, acc))
        print("-" * 80)
    resnet18_ft.train()

# train_x = range(len(train_curve))
# train_y = train_curve
#
# train_iters = len(train_data_loader)
# valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
# valid_y = valid_curve

# plt.plot(train_x, train_y, label='Train')
# plt.plot(valid_x, valid_y, label='Valid')
# plt.legend(loc='upper right')
# plt.ylabel('loss value')
# plt.xlabel('Iteration')
# plt.show()
#
# plt.savefig("./result.png")
