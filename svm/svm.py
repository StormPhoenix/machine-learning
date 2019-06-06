import numpy as np
import matplotlib.pyplot as plt

SVM_TEST_SET_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/svm/testSet.txt'


class SmoCacheStruct:
    def __init__(self, dataMatIn, labels, paramC, paramToler):
        self.x = dataMatIn
        self.y = labels
        self.paramC = paramC
        self.paramToler = paramToler
        self.dataSize = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.dataSize, 1)))
        self.b = np.mat([[0]])
        self.errorCache = np.mat(np.zeros((self.dataSize, 2)))


def calculateError(scs: SmoCacheStruct, i):
    Ui = float(np.multiply(scs.alphas, scs.y).T * (scs.x * scs.x[i, :].T)) + scs.b
    Ei = Ui - float(scs.y[i])
    return Ei


def showGraph(coord, label, x1, y1, x2, y2):
    coordArray = np.array(coord)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coordArray[:, 0], coordArray[:, 1],
               (np.array(label) + 2) * 15, (np.array(label) + 2) * 15)
    plt.plot([x1, x2], [y1, y2])
    plt.show()


def clipAlpha(alpha, low, high):
    if alpha < low:
        alpha = low
    elif alpha > high:
        alpha = high
    return alpha


def updateErrorK(scs, k):
    errorK = calculateError(scs, k)
    scs.errorCache[k] = [1, errorK]


def selectJ(scs: SmoCacheStruct, i, errorI):
    # TODO 提取到其他函数
    scs.errorCache[i] = [1, errorI]

    j = -1
    errorJ = 0
    maxDelta = 0
    validErrorCacheIndex, _ = np.nonzero(scs.errorCache[:, 0])

    if len(validErrorCacheIndex) > 1:
        for k in validErrorCacheIndex:
            if k == i:
                continue
            ## TODO Why do I have to calculate it again?
            errorK = calculateError(scs, k)
            delta = abs(errorK - errorI)
            if delta > maxDelta:
                maxDelta = delta
                errorJ = errorK
                j = k
        # 原代码
        # if j != -1:
        #     return j, errorJ
        # TODO 此处, 书中代码不严谨,修改如下
        if j != -1:
            return j, errorJ
        else:
            return j, calculateError(scs, j)
        # return j, errorJ
    j = selectRandomly(i, scs.dataSize)
    errorJ = calculateError(scs, j)
    return j, errorJ


def selectRandomly(i: int, m: int):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArray = line.strip().split('\t')
        dataMat.append([float(lineArray[0]), float(lineArray[1])])
        labelMat.append(int(lineArray[2]))
    return dataMat, labelMat


def smoPlatt(dataMat, labels, paramC, paramToler, maxIterCount):
    scs: SmoCacheStruct = SmoCacheStruct(np.mat(dataMat), np.mat(labels).transpose(), paramC, paramToler)
    iterCount = 0
    newEntireSet = True
    alphasPairsChanged = 0

    # TODO
    while (iterCount < maxIterCount) and (alphasPairsChanged > 0 or newEntireSet):
        alphasPairsChanged = 0
        if newEntireSet:
            for i in range(scs.dataSize):
                alphasPairsChanged += smoPlattInner(scs, i)
                print("full set, iter: %d i: %d, pairs changed %d", iterCount, alphasPairsChanged)
            iterCount += 1
        else:
            nonBoundAlphasIndex, _ = np.nonzero((scs.alphas.A > 0) * (scs.alphas.A < scs.paramC))
            for i in nonBoundAlphasIndex:
                alphasPairsChanged += smoPlattInner(scs, i)
                print("full set, iter: %d i: %d, pairs changed %d", iterCount, alphasPairsChanged)
            iterCount += 1
        if newEntireSet:
            newEntireSet = False
        elif alphasPairsChanged == 0:
            newEntireSet = True
        print("iteration number: %d" % iterCount)
    return scs.b, scs.alphas


def smoPlattInner(scs: SmoCacheStruct, i):
    errorI = calculateError(scs, i)
    if ((scs.y[i] * errorI < - scs.paramToler and scs.alphas[i] < scs.paramC)
            or (scs.y[i] * errorI > scs.paramToler and scs.alphas[i] > 0)):
        j, errorJ = selectJ(scs, i, errorI)
        oldAlphaI = scs.alphas[i].copy()
        oldAlphaJ = scs.alphas[j].copy()
        # calculate border
        if scs.y[i] != scs.y[j]:
            L = max(0, scs.alphas[j] - scs.alphas[i])
            H = min(scs.paramC, scs.paramC + scs.alphas[j] - scs.alphas[i])
        else:
            L = max(0, scs.alphas[i] + scs.alphas[j] - scs.paramC)
            H = min(scs.paramC, scs.alphas[i] + scs.alphas[j])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * scs.x[i, :] * scs.x[j].T - scs.x[i, :] * scs.x[i, :].T - scs.x[j, :] * scs.x[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return
        scs.alphas[j] -= scs.y[j] * (errorI - errorJ) / eta
        scs.alphas[j] = clipAlpha(scs.alphas[j], L, H)
        # scs.alphas[j] = oldAlphaJ + scs.y[j] * (errorJ - errorI) / eta
        # scs.alphas[j] = clipAlpha(scs.alphas[j], L, H)
        updateErrorK(scs, j)
        if abs(scs.alphas[j] - oldAlphaJ) < 0.00001:
            print("j not moving enough.")
            return 0
        scs.alphas[i] += scs.y[j] * scs.y[i] * (oldAlphaJ - scs.alphas[j])
        updateErrorK(scs, i)
        b1 = (scs.b - errorI
              - scs.y[i] * (scs.alphas[i] - oldAlphaI) * scs.x[i, :] * scs.x[i, :].T
              - scs.y[j] * (scs.alphas[j] - oldAlphaJ) * scs.x[i, :] * scs.x[j, :].T)
        b2 = (scs.b - errorJ
              - scs.y[i] * (scs.alphas[i] - oldAlphaI) * scs.x[i, :] * scs.x[i, :].T
              - scs.y[j] * (scs.alphas[j] - oldAlphaJ) * scs.x[j, :] * scs.x[j, :].T)

        if (0 < scs.alphas[i]) and (scs.alphas[i] < scs.paramC):
            scs.b = b1
        elif (0 < scs.alphas[j]) and (scs.alphas[j] < scs.paramC):
            scs.b = b2
        else:
            scs.b = 0.5 * (b1 + b2)
        return 1
    else:
        return 0


def smoSimple(dataMat, labels, paramC, toler, maxIterCount):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labels).transpose()
    dataSize, featureCount = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((dataSize, 1)))
    b = 0
    iterCount = 0

    while iterCount < maxIterCount:
        changedAlphaPairs: int = 0
        for i in range(dataSize):
            Ui = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = Ui - float(labelMatrix[i])

            if (((labelMatrix[i] * Ei < -toler) and (alphas[i] < paramC))
                    or ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectRandomly(i, dataSize)
                Uj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = Uj - float(labelMatrix[j])
                oldAlphaI = alphas[i].copy()
                oldALphaJ = alphas[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(paramC, paramC + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - paramC)
                    H = min(paramC, alphas[j] + alphas[i])

                if L == H:
                    print("L == H")
                    continue

                eta = (2.0 * dataMatrix[i, :] * dataMatrix[j, :].T -
                       dataMatrix[i, :] * dataMatrix[i, :].T -
                       dataMatrix[j, :] * dataMatrix[j, :].T)
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], L, H)
                if (abs(alphas[j] - oldALphaJ) < 0.00001):
                    print("J not moving enough")
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (oldALphaJ - alphas[j])

                b1 = (b - Ei - labelMatrix[i] * (alphas[i] - oldAlphaI) * dataMatrix[i, :] * dataMatrix[i, :].T
                      - labelMatrix[j] * (alphas[j] - oldALphaJ) * dataMatrix[i, :] * dataMatrix[j, :].T)

                b2 = (b - Ej - labelMatrix[j] * (alphas[i] - oldAlphaI) * dataMatrix[i, :] * dataMatrix[i, :].T
                      - labelMatrix[j] * (alphas[j] - oldALphaJ) * dataMatrix[j, :] * dataMatrix[j, :].T)

                if (0 < alphas[i]) and (alphas[i] < paramC):
                    b = b1
                elif (0 < alphas[j]) and (alphas[j] < paramC):
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                changedAlphaPairs += 1
                print("iter: %d i:%d, pairs changed %d" % (iterCount, i, changedAlphaPairs))
        if changedAlphaPairs == 0:
            iterCount += 1
        else:
            iterCount = 0
        print("iteration number: %d" % iterCount)
    return b, alphas


def main():
    dataMat, labelMat = loadDataSet(SVM_TEST_SET_PATH)
    b, alphas = smoPlatt(dataMat, labelMat, 9, 0.001, 40)
    # b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print(b, alphas[alphas > 0])

    W = np.multiply(np.multiply(alphas, np.mat(labelMat).transpose()), np.mat(dataMat)).sum(0)
    w1 = W.getA1()[0]
    w2 = W.getA1()[1]
    b = b.getA1()[0]
    y1 = -5
    x1 = (-b - w2 * y1) / w1

    y2 = 3
    x2 = (-b - w2 * y2) / w1
    showGraph(dataMat, labelMat, x1, y1, x2, y2)


if __name__ == '__main__':
    main()
