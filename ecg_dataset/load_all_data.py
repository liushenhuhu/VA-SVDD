import numpy as np
from ecg_dataset.load_data.load_data_mitbih import load_mitbih_data
from ecg_dataset.load_data.load_data_cudb import load_cudb_data
from ecg_dataset.load_data.load_data_vfdb import load_vfdb_data

def load_vfdb(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, len_noise =load_vfdb_data(noisetest=opt.noisetest)
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)
    if opt.noisetest==False:
        train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed,
                                                                                         0)
    else:
        train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed,
                                                                                             len_noise)
    return train_data,train_label, val_data, val_label, test_data, test_label

def load_mitbih(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_mitbih_data()
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed,
                                                                                         0)

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_cudb(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_cudb_data()
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed,
                                                                                         0)
    return train_data, train_label, val_data, val_label, test_data, test_label

def get_data_label(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED,len_noise):
    # train_normal_data, test_normal_data, test_abnormal_data = load_data(root)

    len_train = train_data.shape[0]
    len_val_normal = val_normal_data.shape[0]
    len_val_abnormal = val_abnormal_data.shape[0]

    len_test_normal = test_normal_data.shape[0]
    len_test_abnormal = test_abnormal_data.shape[0]

    train_label = np.zeros(len_train)
    val_data = np.concatenate([val_normal_data, val_abnormal_data], axis=0)
    val_label = np.concatenate([np.zeros(len_val_normal), np.ones(len_val_abnormal)], axis=0)

    test_data = np.concatenate([test_normal_data, test_abnormal_data], axis=0)
    test_label = np.concatenate([np.zeros(len_test_normal-len_noise//2),np.full(len_noise//2,2) , np.ones(len_test_abnormal)], axis=0)

    train_label, train_idx = shuffle_label(train_label, SEED)
    train_data = train_data[train_idx]

    val_label, val_idx = shuffle_label(val_label, SEED)
    val_data = val_data[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_label, SEED)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]


    return train_data, train_label, val_data, val_label, test_data, test_label

def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)

    return labels, index


def shuffle_data(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED):
    len_train = train_data.shape[0]
    len_val_normal = val_normal_data.shape[0]
    len_val_abnormal = val_abnormal_data.shape[0]

    len_test_normal = test_normal_data.shape[0]
    len_test_abnormal = test_abnormal_data.shape[0]

    train_label = np.zeros(len_train)
    val_data = np.concatenate([val_normal_data, val_abnormal_data], axis=0)
    val_label = np.concatenate([np.zeros(len_val_normal), np.ones(len_val_abnormal)], axis=0)

    test_data = np.concatenate([test_normal_data, test_abnormal_data], axis=0)
    test_label = np.concatenate([np.zeros(len_test_normal), np.ones(len_test_abnormal)], axis=0)

    train_label, train_idx = shuffle_label(train_label, SEED)
    train_data = train_data[train_idx]

    val_label, val_idx = shuffle_label(val_label, SEED)
    val_data = val_data[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_label, SEED)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]

    return train_data, train_label, val_data, val_label, test_data, test_label