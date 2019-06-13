import numpy as np
import graph.scatter as scatter


def loadDataSet(filename):
    dataMat = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        splitLine = line.strip().split('\t')
        floatsLine = list(map(float, splitLine))
        dataMat.append(floatsLine)
    return dataMat


def main():
    dataSet = loadDataSet('../data/reg/ex00.txt')
    dataMat = np.mat(dataSet)
    scatter.showScatters(dataMat[:, 0].T.A1, dataMat[:, 1].T.A1)


if __name__ == '__main__':
    main()
