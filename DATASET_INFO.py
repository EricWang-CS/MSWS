import scipy.io as scio
import numpy as np
from sklearn import metrics

DS = {1: 'MSRC-v1', 2: 'BBCnews', 3: 'BBCsports', 4: 'MNIST', 5: 'HW', 6: 'Caltech101-7', 7: 'Caltech101-20',
      8: 'Caltech101-all', 9: 'Youtube', 10: 'NUS-WIDE', 11: "Reuters", 12: "NUSWIDEOBJ", 13: 'yale_mtv', 14:'3Source'}
BASE_PATH = "data/"

CHECKs = [14]

for ind in CHECKs:
    DATASET = DS[ind]
    data = scio.loadmat(BASE_PATH + DATASET + ".mat")
    # print(data)
    print("Dataset: ", DATASET)
    print("Dimensions: ", [x.shape[1] for x in data['X'][0]])
    print("Views: ", len(data['X'][0]))
    print("Samples: ", data['X'][0][0].shape)
    print("Class:", np.unique(data['Y']))
    print([np.sum(data['Y'] == i) for i in np.unique(data['Y'])])
    print("*" * 30)
