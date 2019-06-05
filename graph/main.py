import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpt3d
# import matplotlib.axes._subplots.Axes3DSubplot as Axes3D
import numpy as np


def main():
    fig = plt.figure()

    z = np.linspace(0, 13, 1000)
    x = 5 * np.sin(z)
    y = 5 * np.cos(z)
    zd = 13 * np.random.random(100)
    xd = 5 * np.sin(zd)
    yd = 5 * np.cos(zd)
    ax2 = plt.subplot(111, projection='3d')
    # ax2: Axes3D = plt.subplot(111, projection='3d')
    print(type(ax2))
    ax2.scatter3D(xd, yd, zd, cmap='Blue')
    # ax2.scatter3D(x, y, z, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main();
