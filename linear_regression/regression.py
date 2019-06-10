import linear_regression.loader as loader
import matplotlib.pyplot as plt
import graph.scatter as scatter
import numpy as np
import numpy.linalg as linalg

'''
TO DO LIST 仔细学习矩阵求导
'''


def locallyWeightedLinearRegres(predictX, dataArr, labelArr, k=0.01):
    xMat = np.mat(dataArr)
    yMat = np.mat(labelArr).transpose()
    m, _ = np.shape(xMat)
    weights = np.mat(np.eye(m))

    for j in range(m):
        diffMat = predictX - xMat[j, :]
        weights[j, j] = np.exp(np.dot(diffMat, diffMat.T) / (-2.0 * k ** 2))

    xTWx = xMat.T * (weights * xMat)
    if linalg.det(xTWx) == 0:
        print('This matrix is singular, cannot do inverse')
        return

    ws = xTWx.I * (xMat.T * (weights * yMat))
    return predictX * ws


def standRegres(xArr, yArr):
    '''
    纯直线线性回归
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    xTx = xMat.T * xMat

    # 计算行列式
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # 依据求导公式得出 ws,涉及到矩阵求导
    ws = xTx.I * (xMat.T * yMat)
    return ws


def testLinearRegres():
    dataArr, labelArr = loader.loadDataSet()
    ws = standRegres(dataArr, labelArr)
    print('ws:', ws)

    xMat = np.mat(dataArr)
    yMat = np.mat(labelArr).transpose()
    yHat = np.dot(xMat, ws)

    # draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A1, yMat[:, 0].flatten().A1)
    xCopy = xMat[:, 1]
    print(np.shape(xCopy))
    ax.plot(xCopy[:, 0].flatten().A1, yHat[:, 0].flatten().A1)
    plt.show()
    # 计算相关系数
    coef = np.corrcoef(yHat.T, yMat.T)
    print("coef shape", np.shape(coef))
    print("coef: ", coef)


def testLwlr():
    dataArr, labelArr = loader.loadDataSet()
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, _ = np.shape(dataMat)

    xMat = dataMat.copy()
    xMat = xMat[xMat[:, 1].argsort(0)][:, 0, :]
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i] = locallyWeightedLinearRegres(xMat[i, :], dataArr, labelArr)
    # draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1].flatten().A1, labelMat[:, 0].flatten().A1)
    ax.plot(xMat[:, 1].flatten().A1, yHat[:, 0].flatten().A1)
    plt.show()
    # 计算相关系数
    coef = np.corrcoef(yHat.T, labelMat.T)
    print("coef shape", np.shape(coef))
    print("coef: ", coef)


def main():
    # testLinearRegres()
    testLwlr()


if __name__ == '__main__':
    main()
