import numpy as np
import tree_regression.loader as loader


def binSplitDataSet(dataSet, feature, value):
    dataMat = np.mat(dataSet)
    mat0 = dataMat[np.nonzero(dataMat[:, feature] > value)[0], :]
    mat1 = dataMat[np.nonzero(dataMat[:, feature] <= value)[0], :]
    return mat0, mat1


def regErr(dataMat):
    return np.var(dataMat[:, -1]) * np.shape(dataMat)[0]


def regLeaf(dataMat):
    return np.mean(dataMat[:, -1])


def choseBestSplitFeature(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    dataMat = np.mat(dataSet)
    if len(list(set(dataMat[:, -1].T.A1))) == 1:
        return None, leafType(dataMat)

    varStep, minDataSize = ops
    bestFeat = -1
    bestVal = -1
    bestS = np.inf
    m, n = np.shape(dataMat)
    S = errType(dataMat)
    for feat in range(n - 1):
        for splitVal in set(dataMat[:, feat].T.A1):
            mat0, mat1 = binSplitDataSet(dataMat, feat, splitVal)
            if (np.shape(mat0)[0] < minDataSize or
                    np.shape(mat1)[0] < minDataSize):
                continue
            tempError = regErr(mat0) + regErr(mat1)
            if tempError < bestS:
                bestS = tempError
                bestFeat = feat
                bestVal = splitVal

    if S - bestS < varStep:
        return None, leafType(dataSet)
    return bestFeat, bestVal


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feature, val = choseBestSplitFeature(dataSet, leafType, errType, ops)
    if feature is None:
        return val
    mat0, mat1 = binSplitDataSet(dataSet, feature, val)
    tree = {'feature': feature, 'value': val, 'left': createTree(mat0, leafType, errType, ops),
            'right': createTree(mat1, leafType, errType, ops)}
    return tree


def isTree(tree):
    return type(tree).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, crossSet):
    if not isTree(tree):
        return tree
    if np.shape(crossSet)[0] == 0:
        return getMean(tree)

    feature = tree['feature']
    val = tree['value']
    leftMat, rightMat = binSplitDataSet(crossSet, feature, val)

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], leftMat)

    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], leftMat)

    if (not isTree(tree['left'])
            and not isTree(tree['right'])):
        mat0, mat1 = binSplitDataSet(crossSet, feature, val)
        errorUnmerge = (np.sum(np.power(mat0[:, -1] - tree['left'], 2))
                        + np.sum(np.power(mat1[:, -1] - tree['right'], 2)))
        mean = (tree['left'] + tree['right']) / 2.0
        crossMat = np.mat(crossSet)
        errorMerge = np.sum(np.power(crossMat[:, -1] - mean, 2))
        if errorMerge < errorUnmerge:
            print('Merge tree')
            return mean
    return tree


def main():
    dataSet = loader.loadDataSet('../data/reg/ex2.txt')
    dataMat = np.mat(dataSet)
    import graph.scatter as scatter
    scatter.showScatters(dataMat[:, 0].T.A1, dataMat[:, 1].T.A1)
    tree = createTree(dataSet, ops=(0, 1))
    print(tree)
    dataTest = loader.loadDataSet('../data/reg/ex2test.txt')
    prune(tree, dataTest)
    print(tree)


if __name__ == '__main__':
    main()
