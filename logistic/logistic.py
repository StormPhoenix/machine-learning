import numpy as np

PATH = '/home/stormphoenix/Workspace/ai/machine-learning/data/testSet.txt'


def loadDataSet():
    dataSet = []
    dataLabel = []
    fr = open(PATH, 'r')
    for line in fr.readlines():
        splitLine = line.strip().split()
        dataSet.append([1.0, float(splitLine[0]), float(splitLine[1])])
        dataLabel.append(int(splitLine[2]))
    return dataSet, dataLabel


def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


def gradient(dataSet, dataLabel):
    dataMat = np.mat(dataSet)
    labelMat = np.mat(dataLabel).transpose()

    m, n = np.shape(dataMat)
    weights = np.ones((n, 1))
    speed = 0.01
    cyclic = 500
    for i in range(cyclic):
        z = sigmoid(dataMat * weights)
        error = (labelMat - z)
        weights = weights + speed * dataMat.transpose() * error
    return weights


def main():
    dataSet, dataLabel = loadDataSet()
    weights = gradient(dataSet, dataLabel).getA()

    xcoord0 = []
    ycoord0 = []
    xcoord1 = []
    ycoord1 = []

    for i, data in enumerate(dataSet):
        label = dataLabel[i]
        if label == 0:
            xcoord0.append(data[1])
            ycoord0.append(data[2])
        else:
            xcoord1.append(data[1])
            ycoord1.append(data[2])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord0, ycoord0, s=30, c='red', marker='s')
    ax.scatter(xcoord1, ycoord1, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = ((-weights[0] - weights[1] * x) / weights[2])
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    main()
