import numpy as np


import numpy as np

def load_cudb_data(noisetest = False):
    vt = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/vt.npy')
    vt = vt.reshape(-1, 2500)  # 5,2500
    vt = np.nan_to_num(vt)

    vfvfl = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/vfvfl.npy')
    vfvfl = vfvfl[:770000].reshape(-1, 2500)  # 308 2500
    vfvfl = np.nan_to_num(vfvfl)

    va = np.concatenate((vt, vfvfl), axis=0)

    normal = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/normal.npy')
    normal = normal[:49132500].reshape(-1, 2500)
    normal = normal[:942, :]
    normal = np.nan_to_num(normal)

    noisy = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/noisy.npy')
    noisy = noisy.reshape(-1, 2500)
    noisy = noisy[:942, :]
    noisy = np.nan_to_num(noisy)

    len_all = len(noisy) + len(normal)

    train_X = (normal[0:len_all * 2 // 3])

    val_normal_X = np.concatenate((normal[len_all * 2 // 3:-(len(normal)-len_all * 2 // 3)],
                                  noisy[0:len(noisy) * 1 // 2]),axis=0)
    val_abnormal_X = va[0:len(va) // 2]

    test_normal_X = np.concatenate((normal[len_all * 2 // 3:],
                                  noisy[len(noisy) * 1 // 2:]),axis=0)
    test_abnormal_X = va[len(va) // 2:]

    if noisetest == True:
        return train_X, val_normal_X, val_abnormal_X, noisy[len(noisy) * 1 // 2:], []#测试集只要噪声
    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X

