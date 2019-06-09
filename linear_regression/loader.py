import numpy as np

REGRESSION_EX0_PATH = '../data/linearregression/ex0.txt'


def loadDataSet(filepath=REGRESSION_EX0_PATH):
    dataMat = []
    labelMat = []
    featureCount = len(open(filepath, 'r').readline().split('\t')) - 1

    fr = open(filepath, 'r')
    for line in fr.readlines():
        lineArr = []
        splitArr = line.strip().split('\t')
        for feature in range(featureCount):
            lineArr.append(float(splitArr[feature]))
        dataMat.append(lineArr)
        labelMat.append(float(splitArr[-1]))
    return dataMat, labelMat
