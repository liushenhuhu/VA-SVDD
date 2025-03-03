
import os
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score, auc, average_precision_score, f1_score,classification_report,confusion_matrix,precision_recall_fscore_support,precision_recall_curve, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt


# a = 0.3214323
# bb = "%.2f%%" % (a * 100)
# print bb
# # 输出结果是32.14%



def evaluate(labels, scores,res_th=None, saveto=None):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # labels = labels.cpu()
    # scores = scores.cpu()

    # True/False Positive Rates.


    # Equal Error Rate
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # if saveto:
    #     plt.figure()
    #     lw = 2
    #     plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
    #     plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    #     plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic')
    #     plt.legend(loc="lower right")
    #     plt.savefig(os.path.join(saveto, "ROC.pdf"))
    #     plt.close()
    #
    # # best f1
    best_f1 = 0
    best_threshold = 0
    #
    # for threshold in ths:
    #
    #     tmp_scores = scores.copy()
    #     tmp_scores[tmp_scores >= threshold] = 1
    #     tmp_scores[tmp_scores < threshold] = 0
    #     cur_f1 = f1_score(labels, tmp_scores)
    #     if cur_f1 > best_f1:
    #         best_f1 = cur_f1
    #         best_threshold = threshold
    #
    # fpr_intrp = interp1d(ths, fpr)
    #
    # fpr = (fpr_intrp(best_threshold))
    #
    # print('FPR:{}'.format(fpr))


    def get_percentile(scores, normal_ratio):
        per = np.percentile(scores, normal_ratio)
        return per

    if res_th is not None and saveto is  not None:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= res_th] = 1
        tmp_scores[tmp_scores < res_th] = 0
        print(classification_report(labels,tmp_scores))
        print(confusion_matrix(labels,tmp_scores))



    test_normal_ratio = int(len(np.where(labels == 0)[0]) / len(labels) * 100)
    per = get_percentile(scores,test_normal_ratio)
    y_pred = (scores >= per)


    Pre,Recall, f1, _ = precision_recall_fscore_support(labels,y_pred,average='binary')
    #Recall = recall_score(labels.astype(int),scores.astype(int),average='binary')
    # acc = accuracy_score(labels, tmp_scores)
    # con_matrix=confusion_matrix(labels, tmp_scores)

    fpr, tpr, ths = roc_curve(labels, y_pred)

    # acc=accuracy_score(labels,scores)
    roc_auc = auc(fpr, tpr)
    auc_prc = average_precision_score(labels, scores)
    roc_auc=roc_auc_score(labels, scores)
    # return auc_prc,roc_auc,best_threshold,Pre,Recall,f1,labels,y_pred
    return auc_prc,roc_auc,Pre,Recall,f1


