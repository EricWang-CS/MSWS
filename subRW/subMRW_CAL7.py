import math
import numpy as np
import scipy.io as scio
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from utils import random_lebeled_index, chain_matmul
from subRW.SubMarkovRandomWalk import subRW
from subRW.Hypergraph import *

if __name__ == "__main__":
    DATASET = "Caltech101-7"
    C = 5e-1
    LRs = [i * 5 for i in range(1, 11)]
    # kns = [20 for i in range(10)]
    kns = [25,25,20,20,15,15,10,10,5,5]
    for i in range(10):
        LABELED_RATE = LRs[i]  # 20
        K_Neighbors = kns[i]
        METRIC = "cosine"

        data = scio.loadmat("../data/" + DATASET + ".mat")
        Xs = data['X'][0]

        Y = data['Y'].reshape((-1,)).astype(np.int)
        CLASSES = len(np.unique(Y))

        print([(Y == i).sum() for i in range(1, CLASSES + 1)])

        ACCS = []

        print(len(Y))

        for j in range(10):
            unlabeled_index = scio.loadmat(
                "C:/Users/林鹏飞/PycharmProjects/multi_view_semi_supervised/data/" + DATASET + "/" + str(
                    LABELED_RATE) + "PER/" + str(j) + ".mat")['unlabeled_index'].reshape((-1,)).astype(np.int)
            Y_train = Y.copy()
            Y_train[unlabeled_index] = -1

            H = multiview_to_hypergraph(Xs, metric=METRIC, neighbors=K_Neighbors)

            W = hypergraph_weights(H, Y_train)
            P = Probability(H, W)

            labels = subRW(Y_train, CLASSES, C, P)

            # print([(labels[unlabeled_index] == i).sum() for i in range(1, CLASSES + 1)])
            # print([(Y[unlabeled_index] == i).sum() for i in range(1, CLASSES + 1)])
            acc = metrics.accuracy_score(labels[unlabeled_index], Y[unlabeled_index])
            print("No.", i, DATASET, " C:", C, " ACC:", acc)
            ACCS.append(acc)

        with open("BBCnews.txt", "a+", encoding="UTF-8") as f:
            print("Labeled_rate:", LABELED_RATE, "C:", C, "Nei:", K_Neighbors, DATASET, np.mean(np.array(ACCS)))
            f.write(
                "Labeled_rate:" + str(LABELED_RATE) + "\tC:" + str(C) + "\tNei:" + str(K_Neighbors) + "\t" + DATASET + "\t" + str(
                    np.mean(np.array(ACCS))) + "\n")
    with open("BBCnews.txt", "a+", encoding="UTF-8") as f:
        f.write("*"*50+"\n")
