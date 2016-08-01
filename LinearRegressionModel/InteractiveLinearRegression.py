# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import Kernel

kernel = Kernel.gauss
xList = [-0.5, 0.5]
tList = [ 0, 0]
def plotRegression(w):
    plt.clf()
    Xscale = np.linspace(-0.5, 0.5, 1000)
    PHIPlot = kernel(Xscale, M)
    Yscale = np.dot(PHIPlot, w)
    # plot kernel
    for i in range(len(w)):
        plt.plot(Xscale, w[i] * PHIPlot[:, i])
    # plot regression
    plt.plot(Xscale, Yscale, linewidth=3, label="assume regression")
    # plot sample
    plt.scatter(xList, tList, c = "green", label = "sample point")
    plt.legend()
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.draw()


def onclick(event):
    print event.xdata, event.ydata
    xList.append(event.xdata)
    tList.append(event.ydata)
    plotRegression(fit())

def fit():
    N = len(xList)
    x = np.array(xList)
    t = np.array(tList)
    phi = kernel(x, M)
    alpha = 10 ** -10
    PHI = np.dot(np.linalg.inv((alpha * np.eye(M, M) + np.dot(phi.T, phi))), phi.T)
    w = np.dot(PHI, t)
    return w
if __name__ == "__main__":
    M = 50
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plotRegression(fit())
    plt.show()

