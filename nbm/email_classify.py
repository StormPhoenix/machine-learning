import nbm.bayesian as bayes
import random

HAM_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/ham/%d.txt'
SPAM_PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/spam/%d.txt'


def main():
    docList, classLabel, fullText = createDataSet()
    vocabList = bayes.createVocabList(docList)
    testDocList = []
    testLabelList = []
    testSize = 20
    for i in range(testSize):
        randIndex = int(random.uniform(0, len(docList)))
        testDocList.append(docList[randIndex])
        testLabelList.append(classLabel[randIndex])
        del (docList[randIndex])
        del (classLabel[randIndex])

    trainList = []
    for doc in docList:
        docVec = bayes.words2Vector(doc, vocabList)
        trainList.append(docVec)
    prob0Features, prob1Features, probAbusive = bayes.trainNBByTwoClass(trainList, classLabel)

    correctCount = 0
    for testDoc, testLabel in zip(testDocList, testLabelList):
        testDocVec = bayes.words2Vector(testDoc, vocabList)
        predictLabel = bayes.nbClassify(testDocVec, prob0Features, prob1Features, probAbusive)
        # print('testLabel :', testLabel, 'predictLabel', predictLabel)
        if testLabel == predictLabel:
            correctCount += 1
    print(correctCount)
    print(correctCount / float(testSize))


def createDataSet():
    docList = []
    classLabel = []
    fullText = []
    for i in range(1, 25):
        bigText = open(HAM_PATH % i, encoding='ISO-8859-1').read()
        spiltText = bayes.parseText(bigText)
        docList.append(spiltText)
        fullText.extend(spiltText)
        classLabel.append(1)

        bigText = open(SPAM_PATH % i, encoding='ISO-8859-1').read()
        spiltText = bayes.parseText(bigText)
        docList.append(spiltText)
        fullText.extend(spiltText)
        classLabel.append(0)
    return docList, classLabel, fullText


if __name__ == '__main__':
    main()
