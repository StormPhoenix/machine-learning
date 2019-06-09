import linear_regression.loader as loader
import matplotlib.pyplot as plt
import graph.scatter as scatter
import numpy as np
import numpy.linalg as linalg


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


def main():
    dataArr, labelArr = loader.loadDataSet()
    ws = standRegres(dataArr, labelArr)
    print('ws:', ws)

    xMat = np.mat(dataArr)
    yMat = np.mat(labelArr).transpose()
    yHat = np.dot(xMat, ws)

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


if __name__ == '__main__':
    main()
