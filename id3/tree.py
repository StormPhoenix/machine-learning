import math
import operator
import id3.tree_plotter as pt


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
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    featureName = ['no surfacing', 'flippers']
    return dataSet, featureName


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


def main():
    dataSet, featureName = createDataSet()
    # shannonEntropy = calcShannonEntropy(dataSet)
    # bestFeature, _ = calcBestFeature(dataSet)
    tree = createTree(dataSet, featureName)
    pt.createPlot(tree)


if __name__ == '__main__':
    main()
