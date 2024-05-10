import numpy as np
from sklearn import metrics
from sklearn.semi_supervised import _label_propagation
from sklearn.neighbors import kneighbors_graph
import scipy.io as scio
import random


def subRW(Y, K, c, P=-1):
    B = np.zeros((len(Y), K))
    for i in range(1, K + 1):
        B_idx = np.where(Y == i)[0]
        B[B_idx, i - 1] = 1 / len(B_idx)
    E = np.eye(P.shape[0]) - (1 - c) * P

    R = np.matmul(np.linalg.pinv(E), B)
    return R


if __name__ == "__main__":
    C = 1e-4
    # Constants
    # labeled rate
    labeled_rate = 5
    K_neighbors = 10

    # load data
    data = scio.loadmat("../data/MSRC-v1.mat")
    Xs = data['X'][0]

    Y = data['Y'].reshape((-1,)).astype(np.int)
    CLASSES = len(np.unique(Y))

    # select unlabeled index
    unlabeled_idx = random.sample(range(len(Y)), int(len(Y) * (1 - labeled_rate)))
    # for i in range(1, CLASSES + 1):
    #     labeli_idx = list(np.where(Y == i)[0])
    #     unlabeled_idx = unlabeled_idx + random.sample(labeli_idx, int(len(labeli_idx) * (1 - labeled_rate)))
    # set unlabeled labels = -1
    Y_train = Y.copy()
    Y_train[unlabeled_idx] = -1

    for x in Xs:
        print(x.shape, np.unique(Y))
        # predict with Label Propagation
        LP = _label_propagation.LabelPropagation()
        LP.fit(x, Y)
        # print(LP.predict(X[unlabeled_idx]))
        # print(LP.predict(X[unlabeled_idx]))
        # print(Y[unlabeled_idx])
        print("LP ACC:", LP.score(x[unlabeled_idx], Y[unlabeled_idx]))
        # print(Y)
        # subRW

        XX = np.matmul(x.transpose(), x).diagonal()
        XX = np.diag(np.power(XX + 1e-10, -0.5))
        X = np.matmul(x, XX)
        # (Y, K, c, P=-1, X=-1, k_neighbors=3)
        predict_labels = subRW(Y_train, CLASSES, C, X=X,  k_neighbors=K_neighbors)
        # predict_labels = subRW(X, Y_train, CLASSES, C, k_neighbors=K_neighbors)
        # for i in range(len(Y)):
        #     print(Y[i], " : ", predict_labels[i], " : ", Y_train[i])

        print("subRW ACC: ", metrics.accuracy_score(predict_labels[unlabeled_idx], Y[unlabeled_idx]))
        print("-"*100)
        print()