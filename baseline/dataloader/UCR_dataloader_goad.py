import pickle
import numpy as np
import os
from scipy.signal import savgol_filter, wiener

from pykalman import KalmanFilter


def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.01
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    pred_state = np.squeeze(pred_state)
    return pred_state



def Wiener(x):
    x_Wie = []
    print("WienerFiltering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        signal_Wie = wiener(signal,81)

        # WavePlot_Single(signal_sav,'kalman')

        x_Wie.append(signal_Wie)

    x_Wie = np.array(x_Wie)
    x_Wie = np.expand_dims(x_Wie, 1)

    return x_Wie, x_Wie.shape[-1]


def Kalman_1D(x):
    x_Kal = []
    print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = Kalman1D(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal

def get_EpilepticSeizure(dataset_path, dataset_name):
    data = []
    data_x = []
    data_y = []
    f = open('{}/{}/data.csv'.format(dataset_path, dataset_name), 'r')
    for line in range(0, 11501):
        if line == 0:
            f.readline()
            continue
        else:
            data.append(f.readline().strip())
    for i in range(0, 11500):
        tmp = data[i].split(",")
        del tmp[0]
        del tmp[178]
        data_x.append(tmp)
        data_y.append(data[i][-1])
    data_x = np.asfarray(data_x, dtype=np.float32)
    data_y = np.asarray([int(x) - 1 for x in data_y], dtype=np.int64)

    return data_x ,data_y


def one_class_labeling(labels, normal_class:int,seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)


    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx

def one_class_labeling_sz(labels, abnormal_class:int, seed):
    normal_idx = np.where(labels != abnormal_class)[0]
    abnormal_idx = np.where(labels == abnormal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx




def load_trans_data_ucr_affine(args,dataset_name):


    x_train, x_val, y_val, x_test, y_test = load_data(args, dataset_name)


    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    x_train = x_train.squeeze(-1)  #（171，80） （348，82）
    x_val = x_val.squeeze(-1)   #（359，80）
    x_test = x_test.squeeze(-1)  #（360，80）


    print('Filtering transforms ---')

    x_train = Kalman_1D(x_train)
    x_val = Kalman_1D(x_val)
    x_test = Kalman_1D(x_test)

    x_train = x_train.squeeze(1)  #（171，80） （348，82）
    x_val = x_val.squeeze(1)   #（359，80）
    x_test = x_test.squeeze(1)  #（360，80）



    n_train, n_dims = x_train.shape
    rots = np.random.randn(args.n_rots, n_dims, args.d_out)  #（256，80，32） （256 82 32）


    print('Calculating transforms ---')
    #logging.info('Calculating transforms ---')

    x_train = np.stack([x_train.dot(rot) for rot in rots], 2)  #（171，32，256） （348 32 256）
    x_val = np.stack([x_val.dot(rot) for rot in rots], 2)  #（359，32，256）
    x_test = np.stack([x_test.dot(rot) for rot in rots], 2) #（360，32，256）


    ratio_val = 100.0 * (len(np.where(y_val == 0)[0])) / len(np.array(y_val))
    ratio_test = 100.0 * (len(np.where(y_test == 0)[0])) / len(np.array(y_test))

    return x_train, x_val, y_val, x_test, y_test, ratio_val, ratio_test






def load_data(opt, dataset_name):


    nb_dims = 1

    if dataset_name in ['MFPT']:
        data_X = np.load("{}/{}/{}_data.npy".format(opt.data_UCR, dataset_name, dataset_name))
        # (2574,1024)
        data_Y = np.load("{}/{}/{}_label.npy".format(opt.data_UCR, dataset_name, dataset_name))


    elif dataset_name in ['CWRU']:
        with open('{}/{}/{}_data.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle1:
            data_X = pickle.load(handle1)
            # (8768,1024)

        with open('{}/{}/{}_label.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle2:
            data_Y = pickle.load(handle2)

    elif dataset_name in ['EpilepticSeizure']:

        data_X, data_Y = get_EpilepticSeizure(opt.data_UCR, dataset_name)

    elif dataset_name in ['cpsc']:

        data_X = np.load("{}/{}_data_pure_normal.npy".format(opt.data_CPSC, dataset_name))
        # (2574,1024)
        data_Y = np.load("{}/{}_label_pure_normal.npy".format(opt.data_CPSC, dataset_name))

        data_X = data_X[:, 0:1, :]

        data_X = data_X.reshape(-1, 1, 100)

        data_Y = np.repeat(data_Y, 16)

    elif dataset_name in ['zzu_MI']:
        with open('{}/{}_data.pickle'.format(opt.data_ZZU_MI, dataset_name), 'rb') as handle1:
            data_X = pickle.load(handle1)

            data_X = data_X.transpose(0, 2, 1)

            # data_X = data_X[:,:,:]
            # (64139,1000,12)

        with open('{}/{}_label.pickle'.format(opt.data_ZZU_MI, dataset_name), 'rb') as handle2:
            data_Y_5 = pickle.load(handle2)

            # (64139,5)
            data_Y = np.zeros(data_Y_5.shape[0])
            for i in range(data_Y_5.shape[0]):  # 二值化
                if (data_Y_5[i] == [0, 0, 0, 0, 1]).all():
                    data_Y[i] = 0
                else:
                    data_Y[i] = 1

    else:

        train_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TRAIN.tsv')),
                                delimiter='\t')  #
        test_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TEST.tsv')),
                               delimiter='\t')  #

        data_ALL = np.concatenate((train_data, test_data), axis=0)
        data_X = data_ALL[:, 1:]  # (16637,96)
        data_Y = data_ALL[:, 0] - min(data_ALL[:, 0])  # (16637,)

        if dataset_name == 'FordA':
            for i in range(len(data_Y)):
                if data_Y[i] == 2:
                    data_Y[i] = 1



    nb_timesteps = int(data_X.shape[1] / nb_dims)
    input_shape = (nb_timesteps, nb_dims)

    label_idxs = np.unique(data_Y)
    class_stat = {}

    for idx in label_idxs:
        class_stat[idx] = len(np.where(data_Y == idx)[0])

    if opt.normal_idx >= len(label_idxs):
        normal_idx = opt.normal_idx % len(label_idxs)
    else:
        normal_idx = opt.normal_idx


    if dataset_name == 'EpilepticSeizure':
        labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)

    else:
        labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)

    data_N_X = data_X[idx_normal]  # (4187,96)
    data_N_Y = labels_binary[idx_normal]  # (4187,)  1D
    data_A_X = data_X[idx_abnormal]  # (12450,96)
    data_A_Y = labels_binary[idx_abnormal]  # UCR end

    # Split normal samples
    n_normal = data_N_X.shape[0]
    train_X = data_N_X[:(int(n_normal * 0.6)), ]  # train 0.6
    train_Y = data_N_Y[:(int(n_normal * 0.6)), ]

    val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]  # train 0.2
    val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]

    test_N_X = data_N_X[int(n_normal * 0.8):]  # train 0.2
    test_N_Y = data_N_Y[int(n_normal * 0.8):]

    # val_N_X_len = val_N_X.shape[0]
    # test_N_X_len = test_N_X.shape[0]
    data_A_X_len = data_A_X.shape[0]

    ####### 正常与异常不平衡，采用异常全用的原则###########
    #
    val_N_X_len = data_A_X_len // 2
    test_N_X_len = data_A_X_len // 2

    data_A_X_idx = list(range(data_A_X_len))
    val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]



    val_X = np.concatenate((val_N_X, val_A_X))
    val_Y = np.concatenate((val_N_Y, val_A_Y))
    test_X = np.concatenate((test_N_X, test_A_X))
    test_Y = np.concatenate((test_N_Y, test_A_Y))

    print("[INFO] Labels={}, normal label={}".format(label_idxs, opt.normal_idx))
    print("[INFO] Train: normal={}".format(train_X.shape), )
    print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
    print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )


    # Process data
    test_X = test_X.reshape((-1, input_shape[0], input_shape[1]))


    val_X = val_X.reshape((-1, input_shape[0], input_shape[1]))


    train_X = train_X.reshape((-1, input_shape[0], input_shape[1]))

    # if iter == 0:


    print("True train - Train:{}, Test:{}".format(train_X.shape, test_X.shape))
    return train_X, val_X, val_Y, test_X, test_Y




