# coding:utf-8
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np


def BoxMuller():
    N = 10 ** 6
    X = np.random.random(size=[N, 2])
    Z = 2 * X - 1
    r2 = Z[:, 0] ** 2 + Z[:, 1] ** 2
    index = np.where(r2 < 1)
    Z_new = Z[index[0], :]
    r2_new = r2[index]

    y1 = Z_new[:, 0] * (-2 * np.log(r2_new) / r2_new) ** 0.5
    y2 = Z_new[:, 1] * (-2 * np.log(r2_new) / r2_new) ** 0.5
    Y = np.c_[y1, y2]

    x_plot = np.linspace(-5, 5, 1000)
    y_plot = norm.pdf(x_plot)
    plt.subplot(211)
    plt.hist(y1, bins=50, normed=True)
    plt.plot(x_plot, y_plot, linewidth=3)
    plt.subplot(212)
    plt.hist(y2, bins=50, normed=True)
    plt.plot(x_plot, y_plot, linewidth=3)
    plt.show()


def MCMC():
    def loglikelihood(x):
        # return -(x ** 2 + np.log(2 * np.pi)) / 2
        # return -(x ** 2 + np.log(2 * np.pi)) / 2
        return -((x + 3) ** 2 * (x - 3) ** 2) / 4
    x1 = 0
    prev = 0
    mylist = []
    for i in range(10 ** 6):
        x2 = np.random.normal(x1, 1)
        l = loglikelihood(x2)
        if l > prev or np.exp(l - prev) > np.random.random():
            x1 = x2
        mylist.append(x1)

    x_plot = np.linspace(-5, 5, 1000)
    y_plot = norm.pdf(x_plot)
    y_plot = np.exp(loglikelihood(x_plot))
    plt.subplot(211)
    plt.plot(mylist)
    plt.subplot(212)
    # plt.hist(mylist, bins = 50, normed = True)
    plt.hist(mylist, bins=50)
    plt.plot(x_plot, y_plot)

    plt.show()
if __name__ == "__main__":
    MCMC()
