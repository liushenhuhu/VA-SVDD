import numpy as np


def load_mitbih_data():

    # 读取各类数据，数据类型大小
    # pvc = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/mitdb/pvc.npy')
    # pvc = pvc[:25639200].reshape(-1, 3600)  # (7122,3600)  360hz

    vfvfl = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/mitdb/vf_vfl.npy')
    vfvfl = vfvfl.reshape(-1, 3600)  # （472， 3600）  360hz

    # Fusion_of_ventricular_and_normal_beat = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/mitdb/Fusion_of_ventricular_and_normal_beat.npy')
    # Fusion_of_ventricular_and_normal_beat = Fusion_of_ventricular_and_normal_beat[:2887200].reshape(-1,3600)  # （802， 3600）  360hz

    # normal = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/mitdb/normal.npy')
    # normal = normal[:242593200].reshape(-1, 3600) #（67387， 3600）
    # normal = normal[:1699200].reshape(-1, 3600) # (472, 3600)s

    normal_test = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/mitdb/normal_test.npy')
    # normal_test = normal_test[:242593200].reshape(-1, 3600)  # 67387 .3600
    normal_test = normal_test[:5097600].reshape(-1, 3600)  #1416 3600

    data_X_va = vfvfl
    data_X_normal = normal_test

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_va[0:len(data_X_va) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_va[len(data_X_va) // 2:]



    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X