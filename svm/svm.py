import numpy as np
import matplotlib.pyplot as plt

SVM_TEST_SET_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/svm/testSet.txt'


def showGraph(coord, label):
    coordArray = np.array(coord)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coordArray[:, 0], coordArray[:, 1],
               (np.array(label) + 2) * 15, (np.array(label) + 2) * 15)
    plt.show()


def clipAlpha(alpha, low, high):
    if alpha < low:
        alpha = low
    elif alpha > high:
        alpha = high
    return alpha


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
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print(b, alphas[alphas > 0])

    # dataX = [data[0] for data in dataMat]
    # x1 = min(dataX)
    # Ui = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
    # y1 = np.multiply(alphas, labelMat) (dataMat * )
    # x2 = max(dataX)

    showGraph(dataMat, labelMat)


if __name__ == '__main__':
    main()
