import math
import operator
import id3.tree_plotter as pt
import pickle
import os
import datetime


def calcShannonEntropy(dataSet):
    '''
    :param dataSet: two dimension array
    :return: Number
    '''
    # dataSet.shape[0]
    numEntries = len(dataSet)
    labelCount = {}
    for featureVec in dataSet:
        currentLabel = featureVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1

    # for key, value in labelCount:
    shannonEntropy = 0
    for key in labelCount:
        probability = labelCount[key] / float(numEntries)
        shannonEntropy += -probability * math.log(probability, 2)

    return shannonEntropy


def spiltDataSet(dataSet: [[]], featureAxis: int, value):
    '''
    numpy.array can not extend dynamicaly, so use list type is best
    :param dataSet:
    :param featureAxis:
    :param value:
    :return:
    '''
    resultDataSet = []
    for featureVec in dataSet:
        if featureVec[featureAxis] == value:
            reduceFeature = featureVec[0:featureAxis]
            reduceFeature.extend(featureVec[featureAxis + 1:])
            resultDataSet.append(reduceFeature)
    return resultDataSet


def createDataSet():
    fr = open(DATA_SET_PATH, "r")
    dataSet = [sample.strip().split('\t') for sample in fr.readlines()]
    featureNames = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, featureNames


def calcBestFeature(dataSet: [[]]):
    dataSetSize = len(dataSet)
    featureCounts = len(dataSet[0]) - 1

    bestEntropyDelta: float = 0
    baseEntropy: float = calcShannonEntropy(dataSet)
    bestFeature: int = -1

    for i in range(featureCounts):
        featureValList = [sample[i] for sample in dataSet]
        featureValList = set(featureValList)
        newEntropy: float = 0
        for featureVal in featureValList:
            subDataSet = spiltDataSet(dataSet, i, featureVal)
            probability = len(subDataSet) / float(dataSetSize)
            newEntropy += probability * calcShannonEntropy(subDataSet)
        tempEntropyDelata: float = baseEntropy - newEntropy
        if (tempEntropyDelata > bestEntropyDelta):
            bestFeature = i
            bestEntropyDelta = tempEntropyDelata
    return bestFeature, baseEntropy - bestEntropyDelta


def majorityCount(dataSet):
    labelCounts = {}
    for data in dataSet:
        labelCounts[data[-1]] = labelCounts.get(data[-1], 0) + 1
    sortedLabels = sorted(labelCounts.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabels[0][0]


def createTree(dataSet, featureNames: []):
    labels = [sample[-1] for sample in dataSet]
    if len(labels) == labels.count(labels[0]):
        '''labels' elements are same.'''
        return labels[0]

    if len(dataSet[0]) == 1:
        '''no features avaliable.'''
        return majorityCount(dataSet)

    bestFeature, _ = calcBestFeature(dataSet)
    bestFeatureName = featureNames[bestFeature]
    del (featureNames[bestFeature])

    tree = {bestFeatureName: {}}

    uniqueVals = set([sample[bestFeature] for sample in dataSet])
    for featureVal in uniqueVals:
        subDataSet = spiltDataSet(dataSet, bestFeature, featureVal)
        subFeatureNames = featureNames[:]
        tree[bestFeatureName][featureVal] = createTree(subDataSet, subFeatureNames)
    return tree;


def id3classify(inputTree, featureLabels, testVec):
    featureName = list(inputTree.keys())[0]
    featureDict = inputTree[featureName]

    featureIndex = featureLabels.index(featureName)
    for key in featureDict.keys():
        if key == testVec[featureIndex]:
            if type(featureDict[key]).__name__ == 'dict':
                classLabel = id3classify(featureDict[key], featureLabels, testVec)
            else:
                classLabel = featureDict[key]
    return classLabel


def storeTree(inputTree, fileName):
    fw = open(fileName, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


def loadTree(fileName):
    fr = open(fileName, "rb")
    return pickle.load(fr)


STORE_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/model/id3.model'
DATA_SET_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/lenses.txt'


def main():
    if os.path.isfile(STORE_PATH):
        tree, featureNames = loadTree(STORE_PATH)
    else:
        dataSet, featureNames = createDataSet()
        tree = createTree(dataSet, featureNames[:])
        storeTree((tree, featureNames), STORE_PATH)
    print(id3classify(tree, featureNames, ['young', 'myope', 'no', 'reduced']))
    pt.createPlot(tree)


if __name__ == '__main__':
    main()
