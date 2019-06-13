import numpy as np
import tree_regression.loader as loader
import tree_regression.common as common


def regErr(dataMat):
    return np.var(dataMat[:, -1]) * np.shape(dataMat)[0]


def regLeaf(dataMat):
    return np.mean(dataMat[:, -1])


def main():
    dataSet = loader.loadDataSet('../data/reg/ex2.txt')
    dataMat = np.mat(dataSet)
    import graph.scatter as scatter
    scatter.showScatters(dataMat[:, 0].T.A1, dataMat[:, 1].T.A1)
    tree = common.createTree(dataSet, regLeaf, regErr, ops=(0, 1))
    print(tree)
    dataTest = loader.loadDataSet('../data/reg/ex2test.txt')
    common.prune(tree, dataTest)
    print(tree)


if __name__ == '__main__':
    main()
