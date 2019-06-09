import numpy as np
from graph import scatter as scatter
import boosting.loader as loader
import boosting.boost as boost
import math


def adaBoostTrain(dataArr, classLabels, numIter=40):
    m, _ = np.shape(dataArr)
    D = np.mat(np.ones((m, 1)) / m)
    weakClassifyArr = []
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(numIter):
        # classEst is the prediction result
        bestStump, error, classEst = boost.buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        alpha = float(0.5 * math.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassifyArr.append(bestStump)
        print("classEst: ", classEst.T)
        # - yi * alpha * Gm-1(xi)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = float(aggErrors.sum()) / m

        print("Total error rate: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassifyArr


def main():
    dataMat, labels = loader.loadTestData()
    weakClassifyArr = adaBoostTrain(dataMat, labels, 9)
    print(weakClassifyArr)
    # scatter.showScatterGraph(dataMat, labels)


if __name__ == '__main__':
    main()
