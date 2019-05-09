import operator as op
import os

import matplotlib.pyplot as plt
import numpy as np


def img2Vector(filename):
    returnVal = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVal[0, i * 32 + j] = int(line[j])
    return returnVal


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals
    dataSize = dataSet.shape[0]

    normData = dataSet - np.tile(minVals, (dataSize, 1))
    normData = normData / np.tile(ranges, (dataSize, 1))

    return normData, ranges, minVals


def file2Matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    lineNum = len(lines)

    resultMatrix = np.zeros((lineNum, 3))
    resultLabel = []

    index = 0
    for line in lines:
        line = line.strip()
        splitLine = line.split('\t')
        resultMatrix[index, :] = splitLine[0:3]
        resultLabel.append(int(splitLine[-1]))
        index += 1
    return resultMatrix, resultLabel


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = np.array(['A', 'A', 'B', 'B'])
    return group, label


def kNNClassify(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistDndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistDndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def caseHandwritingClassify():
    trainingLabel = []
    trainingFileList = os.listdir('/home/stormphoenix/Workspace/ai/machine-learning/data/digits/trainingDigits')
    trainingMat = np.zeros((len(trainingFileList), 1024))
    for i, fileNameStr in enumerate(trainingFileList):
        fileName = fileNameStr.split('.')[0]
        number = int(fileName.split('_')[0])
        trainingLabel.append(number)
        trainingMat[i, :] = img2Vector(
            '/home/stormphoenix/Workspace/ai/machine-learning/data/digits/trainingDigits/%s' %
            fileNameStr)

    testFileList = os.listdir('/home/stormphoenix/Workspace/ai/machine-learning/data/digits/testDigits')
    testSetSize = len(testFileList)
    errorNum = 0
    for i, fileNameStr in enumerate(testFileList):
        testVector = img2Vector('/home/stormphoenix/Workspace/ai/machine-learning/data/digits/testDigits/%s' %
                                fileNameStr)
        trueLabel = int(fileNameStr.split('_')[0])
        label = kNNClassify(testVector, trainingMat, trainingLabel, 3)
        if trueLabel != label:
            errorNum += 1
    # TODO analyze the error case
    print(errorNum)
    print(testSetSize)
    print(errorNum / float(testSetSize))


def showDateFigure():
    resultMatrix, resultLabel = file2Matrix('/home/stormphoenix/Workspace/ai/machine-learning/data/datingTestSet.txt')
    normData, ranges, minVals = autoNorm(resultMatrix)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normData[:, 0], normData[:, 1],
               np.array(resultLabel) * 15, np.array(resultLabel) * 15)
    plt.show()


def main():
    # showDateFigure()
    caseHandwritingClassify()


if __name__ == '__main__':
    main()
