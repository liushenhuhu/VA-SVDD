import os
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support, precision_recall_curve, recall_score, accuracy_score



def evaluate_resnet(labels, scores):
    _, predicted = torch.max(scores, 1)  # 获取行最大值下标  0:x 1:noise 2:va
    correct_num = (predicted == labels).squeeze().cpu().sum()

    return correct_num / len(scores) # acc


def evaluate(labels, scores, res_th=None, saveto=None,noisetest=False):

    scores = torch.from_numpy(scores)
    def get_percentile(scores, normal_ratio):
        per = np.percentile(scores, normal_ratio)
        return per
    test_normal_ratio = int(np.count_nonzero(labels !=1) / len(labels) * 100)
    # test_normal_ratio = int(len(np.where(labels == 0)[0]) / len(labels) * 100)
    per = get_percentile(scores, test_normal_ratio)
    # y_pred = [int(x) for x in scores >= per]
    y_pred = (scores >= per)
    if noisetest==True:
        y_pred =np.array(y_pred)
        all_true = 0
        all_false = 0
        noise_true = 0
        noise_false = 0
        for x, y in zip(y_pred,labels):
            x = int(x)

            if x == y:
                all_true += 1
            elif x == 0 and y == 2:
                all_true += 1
                noise_true +=1
            else:
                all_false += 1
                if y == 2:
                    noise_false += 1
        Pre = all_true / (all_true+all_false)

        Pre_noise = (noise_true / (noise_true+noise_false)) if noise_true!=0 else 0
        # print("预测正确数：{} 预测错误数：{}".format(all_true,all_false))
        # print("噪声数据:\n预测正确数：{} 预测错误数：{}".format(noise_true,noise_false))
        return Pre,Pre_noise

    Pre, Recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average='binary')

    auc_prc = average_precision_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)

    roc_auc = auc(fpr, tpr)
    return auc_prc, roc_auc, Pre, Recall, f1, per

