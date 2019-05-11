import numpy as np
import math
import re


def parseText(bigText):
    spiltTestList = re.split(r'\W+', bigText)
    return [token.lower() for token in spiltTestList if len(token) > 2]


def trainNBByTwoClass(trainSet: [[]], trainLabel: []):
    trainSetSize = len(trainSet)
    vecLen = len(trainSet[0])
    probAbusive = sum(trainLabel) / float(trainSetSize)
    label1VecFrequent = np.ones(vecLen)
    label1Sum = 2
    label0VecFrequent = np.ones(vecLen)
    label0Sum = 2
    for i, docVec in enumerate(trainSet):
        if trainLabel[i] == 1:
            label1VecFrequent += docVec
            label1Sum += sum(docVec)
        else:
            label0VecFrequent += docVec
            label0Sum += sum(docVec)
    prob1Features = label1VecFrequent / label1Sum
    prob0Features = label0VecFrequent / label0Sum
    return np.log(prob0Features), np.log(prob1Features), probAbusive


def createVocabList(dataSet: [[]]):
    retSet = set([])
    for data in dataSet:
        retSet = retSet | set(data)
    return list(retSet)


def words2VectorBag(words: [], vocabList: []):
    wordVector = [0] * len(vocabList)
    for word in words:
        if word in vocabList:
            wordVector[vocabList.index(word)] += 1
        else:
            print("the word %s is not in vocabulary" % word)
    return wordVector


def words2Vector(words: [], vocabList: []):
    wordVector = [0] * len(vocabList)
    for word in words:
        if word in vocabList:
            wordVector[vocabList.index(word)] = 1
        else:
            print("the word %s is not in vocabulary" % word)
    return wordVector


def createDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def nbClassify(inputVector, prob0Vec, prob1Vec, prob1Label):
    p1 = np.sum(prob1Vec * inputVector) + math.log(prob1Label)
    p0 = np.sum(prob0Vec * inputVector) + math.log(1 - prob1Label)
    # print('Predict probability : ', p1, p0)
    if p1 > p0:
        return 1
    else:
        return 0


def main():
    postList, classLabel = createDataSet()
    vocabList: [] = createVocabList(postList)
    trainSet = []
    for doc in postList:
        wordVector = words2VectorBag(doc, vocabList)
        # wordVector = words2Vector(doc, vocabList)
        trainSet.append(wordVector)

    testEntry = ['stupid', 'dog', 'dalmation']
    testVector = words2VectorBag(testEntry, vocabList)
    prob0Features, prob1Features, probAbusive = trainNBByTwoClass(trainSet, classLabel)
    predictLabel = nbClassify(testVector, prob0Features, prob1Features, probAbusive)
    # print(vocabList)
    # print(predictLabel)


if __name__ == '__main__':
    main()
