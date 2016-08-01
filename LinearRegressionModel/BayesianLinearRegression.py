# coding:utf - 8
import Kernel
import matplotlib.pyplot as plt
import numpy as np


class BayesianLinerRegression(object):
    pass

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


def myFunc(x):
    return np.sin(2 * np.pi * x) + np.sin(2 * np.pi * x * 3) + np.sin(2 * np.pi * x * 7) - np.sin(2 * np.pi * x * 6)


if __name__ == "__main__":
    # np.random.seed(1)
    N = 100
    M = 30
    kernel = Kernel.gauss
    # サンプルデータ作成
    x = np.random.uniform(-0.5, 0.5, N)
    x = np.sort(x)
    y = myFunc(x)
    t = y + np.random.normal(size=N) * 0.1
    # 計画行列の作成
    phi = kernel(x, M)
    alpha = 10 ** -3
    beta = 1
    # 重みベクトルの推定
    PHI = np.dot(np.linalg.inv((alpha * np.eye(M, M) + np.dot(phi.T, phi))), phi.T)
    w = np.dot(PHI, t)
    Sn = np.linalg.inv(alpha * np.eye(M) + beta * np.dot(phi.T, phi))
    mN = beta * np.dot(np.dot(Sn, phi.T), t)

    z = []
    yPlot = np.linspace(-3, 3, 1000)
    for xPlot in np.linspace(-0.5, 0.5, 1000):
        p = kernel(xPlot, M).T
        m = np.dot(mN, p)[0]
        S = 1 / beta + np.dot(np.dot(p.T, Sn), p)[0]
        z.append(1 / (2 * np.pi * S) ** 0.5 * np.exp(-(yPlot - m) ** 2 / (2 * S)))
    z = np.array(z)
    plt.imshow(z.T, aspect="auto", origin="lower", extent=[-0.5, 0.5, np.min(yPlot), np.max(yPlot)],
               interpolation="nearest")

    Xscale = np.linspace(-0.5, 0.5, 1000)
    PHIPlot = kernel(Xscale, M)
    Yscale = np.dot(PHIPlot, w)
    plt.plot(Xscale, Yscale, linewidth=3, label="assume regression")
    # plot original
    plt.plot(Xscale, myFunc(Xscale), alpha=0.5, linewidth=3, label="original")
    # plot kernel
    """
    for i in range(len(w)):
        plt.plot(Xscale, w[i] * PHIPlot[:, i])
    """
    # plot sample

    plt.scatter(x, t, c="green", label="sample point")
    plt.xlim([-0.5, 0.5])
    plt.ylim([np.min(yPlot), np.max(yPlot)])

    plt.legend()
    plt.show()
