import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

import linear_regression.loader as loader

'''
TO DO LIST 仔细学习矩阵求导
'''


def lwlr(predictX, dataArr, labelArr, k=0.01):
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


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def linearRegresTest(testArr, xArr, yArr):
    ws = standRegres(xArr, yArr)
    print('ws:', ws)
    testMat = np.mat(testArr)
    return np.dot(testMat, ws)


def lwlrTest(testArr, xArr, yArr, k=0.01):
    m = len(testArr)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def main():
    dataArr, labelArr = loader.loadDataSet()
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    m, _ = np.shape(dataMat)
    xMat = dataMat.copy()
    xMat = xMat[xMat[:, 1].argsort(0)][:, 0, :]

    # 测试
    # yHat = linearRegresTest(xMat, dataArr, labelArr)
    yHat = lwlrTest(xMat, dataArr, labelArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1].flatten().A1, labelMat[:, 0].flatten().A1)
    ax.plot(xMat[:, 1].flatten().A1, yHat[:, 0].flatten().A1)
    plt.show()
    # 计算相关系数
    coef = np.corrcoef(yHat.T, labelMat.T)
    print("coef shape", np.shape(coef))
    print("coef: ", coef)


if __name__ == '__main__':
    main()
