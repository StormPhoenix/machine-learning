import numpy as np
import matplotlib.pyplot as plt


def showScatters(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    plt.show()


def showScattersByLabel(x, y, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, (np.array(label) + 2) * 15, (np.array(label) + 2) * 15)
    plt.show()
