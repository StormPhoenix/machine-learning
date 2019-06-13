import numpy as np
import numpy.linalg as linalg
import tree_regression.loader as loader
import tree_regression.common as common


def linearSolve1(dataSet):
    dataMat = np.mat(dataSet)
    m, n = np.shape(dataMat)

    xMat = np.mat(np.ones((m, n)))
    xMat[:, 1:n] = dataMat[:, 0: n - 1]
    yMat = dataMat[:, -1]

    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        raise NameError('This matrix is singular, cannot do inverse, \ntry increasing the second value of ops.')
    ws = xTx.I * (xMat.T * yMat)
    return ws, xMat, yMat


def modelLeaf1(dataSet):
    w, _, _ = linearSolve(dataSet)
    return w


def modelErr1(dataSet):
    dataMat = np.mat(dataSet)
    ws, x, y = linearSolve(dataSet)
    return np.sum(np.power(x * ws - y, 2))


def linearSolve(dataSet):  # helper function used in two places
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)));
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1];
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def main():
    dataSet = loader.loadDataSet('../data/reg/exp2.txt')
    dataTest = loader.loadDataSet('../data/reg/ex2test.txt')
    # import graph.scatter as scatter
    # scatter.showScatters(dataMat[:, 0].T.A1, dataMat[:, 1].T.A1)
    tree = common.createTree(dataSet=dataSet, leafType=modelLeaf1, errType=modelErr1, ops=(0.01, 10))
    print(tree)
    # prune(tree, dataTest)
    # print(tree)


if __name__ == '__main__':
    main()
