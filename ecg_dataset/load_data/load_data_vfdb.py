import numpy as np


def load_vfdb_data(noisetest = False):
    VT = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/VT_test.npy')
    VT = VT[:2617500].reshape(-1, 2500)  # 1047 2500

    vfl = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfl_test.npy')
    vfl = vfl[:347500].reshape(-1, 2500)  # 139 2500

    vfib = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfib_test.npy')
    vfib = vfib[:312500].reshape(-1, 2500)  # 125  2500

    data_X_va = np.concatenate((vfl, vfib, VT), axis=0)

    normal = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/normal_test.npy')
    normal = normal[:8612500].reshape(-1, 2500)  # 3445  2500

    noisy = np.load('/8t/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/noisy_test.npy')
    noisy = noisy[:3030000].reshape(-1, 2500)  # 1212  2500

    len_all = len(noisy)+len(normal)
    print("normal:",len(normal))
    print("noise:",len(noisy))
    print("ab:",len(data_X_va))

    # 噪声平均分配
    # train_X = np.concatenate(((normal[0:len(normal) * 2 // 3]), (noisy[0:len(noisy) * 2 // 3])), axis=0)
    # val_normal_X = np.concatenate((
    #     normal[len(normal) * 2 // 3:len(normal) * 5 // 6], noisy[len(noisy) * 2 // 3:len(noisy) * 5 // 6]
    # ), axis=0)
    # val_abnormal_X = data_X_va[0:len(data_X_va) // 2]
    # test_normal_X = np.concatenate((
    #     normal[len(normal) * 5 // 6:], noisy[len(noisy) * 5 // 6:]
    # ), axis=0)
    # test_abnormal_X = data_X_va[len(data_X_va) // 2:]


    # 原分配方案
    train_X = (normal[0:len_all * 2 // 3])
    val_normal_X = np.concatenate((normal[len_all * 2 // 3:-(len(normal)-len_all * 2 // 3)//2],
                                  noisy[0:len(noisy) * 1 // 2]),axis=0)
    val_abnormal_X = data_X_va[0:len(data_X_va) // 2]
    test_normal_X = np.concatenate((normal[len_all * 2 // 3 + (len(normal)-len_all * 2 // 3)//2:],
                                  noisy[len(noisy) * 1 // 2:]),axis=0)
    test_abnormal_X = data_X_va[len(data_X_va) // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X,len(noisy)


if __name__ == '__main__':
    load_vfdb_data()