import math
import numpy as np
import scipy.io as scio
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from utils import random_lebeled_index, chain_matmul
from subRW.SubMarkovRandomWalk import subRW


def multiview_to_hypergraph(Xs, metric='cosine', neighbors=3):
    H = []
    for x in Xs:
        H.append(kneighbors_graph(x, metric=metric, n_neighbors=neighbors + 1, include_self=True).A)
    # return np.hstack(H)#.transpose()
    return np.vstack(H).transpose()


def multiview_to_hypergraph_finch(Xs, metric='cosine', neighbors=3):
    A = np.zeros((Xs[0].shape[0], Xs[0].shape[0]))
    for x in Xs:
        A = A + kneighbors_graph(x, metric=metric, n_neighbors=neighbors).A
    A = (A != 0).astype(np.int)
    print(A)
    return A.transpose()

    # for x in Xs:
    #     A = kneighbors_graph(x, metric=metric, n_neighbors=neighbors).A
    #     A = (np.matmul(A, A.transpose()) != 0).astype(np.int)
    #     A = A + A.transpose()
    #     H.append((np.matmul(A, A.transpose()) != 0).astype(np.int))
    # return np.vstack(H).transpose()


def hypergraph_weights(H, Y_train):
    print("Compupte weights of hyteredges")
    e = H.shape[1]  # number of hyperedges
    W = np.zeros(e)  # weights
    c = np.max(Y_train)  # max label
    classDict = np.zeros((c, 1))  # number of nodes belonging to class i

    for i in range(c):
        classDict[i] = (Y_train == i + 1).sum()

    edgeClassDict = np.zeros((c, 1))  # number of nodes belonging to class j in edge i

    total_w = 0.0
    count = 0
    for i in range(e):
        Y_K = Y_train * (H[:, i] == 1).reshape((-1,))

        if Y_K.sum() == 0:
            W[i] == -1
            continue

        for j in range(c):
            edgeClassDict[j, 0] = (Y_K == j + 1).sum()
        edgeClassDictRatio = edgeClassDict / classDict
        yPos = edgeClassDictRatio.argmax(axis=0)[0]

        if (max(edgeClassDictRatio) != 0):
            yNegLengthk = edgeClassDict.sum() - edgeClassDict[
                yPos, 0]  # number of instances in that edge with other labels (don't change logic... think deep :P)
        else:
            yNegLengthk = 0
        numNegInstances = classDict.sum() - classDict[yPos, 0]  # Number of negative instances overall

        a = math.pow(edgeClassDict[yPos, 0] / classDict[yPos, 0], 0.5) - math.pow(yNegLengthk / numNegInstances, 0.5)
        hellingerSimilarity = a * a
        W[i] = hellingerSimilarity
        total_w += W[i]
        count += 1
    W = (W==-1)*(total_w/count)+(W!=-1)*W
    print("Compute Weights done!")
    # print(W)
    return np.diag(W)


def Probability(H, W):
    HW = np.matmul(H, W)
    D_v_inv = np.linalg.pinv(np.diag(HW.sum(axis=1)))
    D_e_inv = np.linalg.pinv(np.diag((H.sum(axis=0) - 1)))
    args = [D_v_inv, HW, D_e_inv, H.transpose()]
    P = chain_matmul(args)
    return P  # / P.sum(axis=1).reshape((-1, 1))
