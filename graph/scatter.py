import numpy as np
import matplotlib.pyplot as plt


def showScatterGraph(coord, label):
    coordArray = np.array(coord)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coordArray[:, 0], coordArray[:, 1],
               (np.array(label) + 2) * 15, (np.array(label) + 2) * 15)
    plt.show()
