import numpy as np


def loadTestData():
    dataMat = np.mat([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, labels
