import math

import scipy.io as scio
import numpy as np
import torch
import random
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import Normalizer, StandardScaler


def chain_matmul(mats):
    if len(mats) == 0:
        return np.array([])
    ans_mat = mats[0]
    for i in range(1, len(mats)):
        ans_mat = np.matmul(ans_mat, mats[i])
    return ans_mat


def random_lebeled_index(Y, labeled_rate=2e-1, average=False):
    Y_train = Y.reshape((-1,)).astype(np.int)
    class_num = len(np.unique(Y))
    labeled_index = []
    for i in range(1, class_num+1):
        cur_index = np.where(Y_train==i)[0]
        cur_labeled_index = random.sample(list(cur_index), int(len(cur_index)*labeled_rate))
        # print(len(cur_labeled_index))
        if len(cur_labeled_index) > 0:
            labeled_index.append(cur_labeled_index)
            # print(cur_labeled_index)
    labeled_index = np.hstack(labeled_index)
    unlabeled_indicator = np.ones(len(Y_train))
    unlabeled_indicator[labeled_index] = 0
    unlabeled_index = np.where(unlabeled_indicator == 1)[0]
    return labeled_index, unlabeled_index


class DataLoader(object):
    ALOI = "ALOI"
    CAL_7 = "Caltech101-7"
    CAL_20 = "Caltech101-20"
    HW = "HW"
    MNIST = "MNIST"
    MSRC = "MSRC-v1"
    NUS_WIDE = "NUS-WIDE"
    YOUTUBE = "Youtube"
    ORL = "ORL_mtv"
    BASE_DIR = "C:/Users/林鹏飞/PycharmProjects/multi_view_semi_supervised/data/"

    def __init__(self, dataset):
        data = scio.loadmat(self.BASE_DIR + dataset + ".mat")
        self.X = data['X'][0]

        for i in range(len(self.X)):
            print(self.X[i])

        self.Y = data['Y'].reshape((-1,))
        # print(np.unique(self.Y))
        # print(np.max(self.Y))
        self.Y = self.Y - 1

        self.labels = np.zeros((self.X[0].shape[0], len(np.unique(self.Y))))
        self.labels[:, self.Y] = 1
        self.ITEMS = self.X[0].shape[0]
        self.data = dataset

    def train_data(self, per=2e-1, train_idx=1):
        BASE = "C:/Users/林鹏飞/PycharmProjects/multi_view_semi_supervised/data/"
        PACK = BASE + self.data + "/" + str(per) + "PER"
        labeled_idx = scio.loadmat(PACK + "/" + str(train_idx) + ".mat")['train_idx'].reshape((-1,))
        train_data = np.zeros(self.labels.shape)
        train_data[train_idx, self.Y[labeled_idx]] = 1

        labeled = np.ones(self.Y.shape)
        labeled[labeled_idx] = 0
        unlabeled_idx = np.argwhere(labeled == 1).reshape((-1,))
        return train_data, labeled_idx, unlabeled_idx

    def acc(self, pred_Y, train_idx):
        count = 0
        acc = 0
        train_set = set(train_idx.tolist())
        for i in range(len(self.Y)):
            if i not in train_set:
                count += 1
                if self.Y[i] == pred_Y[i]:
                    acc += 1
        return acc / count


def torch_matmul(mats):
    if len(mats) == 1:
        return mats[0]
    ans = torch.eye(mats[0].shape[0])
    for m in mats:
        ans = torch.matmul(ans, m)
    return ans


def Non0_tensor(X, bias=1e-10):
    return (X != 0).type(torch.float32) * X + (X == 0).type(torch.float32) * 1e-5


def laplacian(X, K=3, mode="knn"):
    if mode == "knn":
        KNG = kneighbors_graph(X, n_neighbors=K, metric="cosine").A
        A = ((KNG + KNG.transpose()) != 0).astype(np.int)
        A = A * (1 - pairwise_distances(X, metric="cosine"))
        return A
    elif mode == "finch":
        KNG = kneighbors_graph(X, n_neighbors=K, metric="cosine").A
        FIN_G = np.matmul(KNG, KNG.transpose())
        A = ((FIN_G + FIN_G.transpose()) != 0).astype(np.int)
        return A


if __name__ == '__main__':
    import os
    import random

    PERCENT = 2e-1
    BASE = "C:/Users/林鹏飞/PycharmProjects/multi_view_semi_supervised/data/"
    DATASET = "ORL_mtv"
    PACK = BASE + DATASET + "/" + str(PERCENT) + "PER"
    print(PACK)
    TOTAL = 50
    if not os.path.exists(BASE + DATASET):
        os.mkdir(BASE + DATASET)
    if not os.path.exists(PACK):
        os.mkdir(PACK)
    data = scio.loadmat(BASE + DATASET)

    samples = data['truth'].shape[0]

    for i in range(TOTAL):
        train_idx = np.array(random.sample(range(samples), int(samples * PERCENT)))
        scio.savemat(PACK + "/" + str(i) + ".mat", {"train_idx": train_idx})


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
