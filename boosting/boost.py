import numpy as np
import math
import boosting.loader as loader


def stumpClassify(dataMatrix, dimen: int, threshVal: float, threshIneq: str):
    retArr = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArr[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArr[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArr


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels).transpose()

    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))

    minError = math.inf

    for i in range(n):
        featureMin = dataMatrix[:, i].min()
        featureMax = dataMatrix[:, i].max()
        stepSize = (featureMax - featureMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (featureMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMatrix] = 0
                # 如果 D 没有归一化,则
                # weightedError = np.dot(D.T, errArr) / np.sum(D)
                weightedError = np.dot(D.T, errArr)

                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                      % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst


def main():
    dataMat, labels = loader.loadTestData()
    m, n = np.shape(dataMat)
    D = np.mat(np.ones((m, 1)) / m)
    bestStump, minError, bestClassEst = buildStump(dataMat, labels, D)
    print(bestStump)
    print(minError)
    print(bestClassEst)


if __name__ == '__main__':
    main()
