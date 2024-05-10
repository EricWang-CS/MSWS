import os
import random
import numpy as np
import scipy.io as scio


def random_lebeled_index(Y, labeled_rate=2e-1, average=False):
    Y_train = Y.reshape((-1,)).astype(np.int)
    class_num = len(np.unique(Y))
    labeled_index = []
    for i in range(1, class_num + 1):
        cur_index = np.where(Y_train == i)[0]
        cur_labeled_index = random.sample(list(cur_index), int(len(cur_index) * labeled_rate))
        if len(cur_labeled_index) > 0:
            labeled_index.append(cur_labeled_index)
    labeled_index = np.hstack(labeled_index)
    unlabeled_indicator = np.ones(len(Y_train))
    unlabeled_indicator[labeled_index] = 0
    unlabeled_index = np.where(unlabeled_indicator == 1)[0]
    return labeled_index, unlabeled_index


DATASET = "ORL_mtv"

for w in range(1, 11):
    PERCENT = w * 5
    BASE = "data/"
    PACK = BASE + DATASET + "/" + str(PERCENT) + "PER"
    # print(PACK)
    TOTAL = 10
    if not os.path.exists(BASE + DATASET):
        os.mkdir(BASE + DATASET)
    if not os.path.exists(PACK):
        os.mkdir(PACK)
    data = scio.loadmat(BASE + DATASET + ".mat")

    samples = data['truth'].astype(np.int).reshape((-1,))
    print(np.unique(data['truth']))
    for x in data['X'][0]:
        print(x.shape)

    for i in range(TOTAL):
        # print(samples)
        li, unlabeled_index = random_lebeled_index(samples, labeled_rate=PERCENT / 100, average=True)
        # print(len(unlabeled_index))
        scio.savemat(PACK + "/" + str(i) + ".mat", {"unlabeled_index": unlabeled_index})
