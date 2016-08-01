# coding:utf-8
# 線形基底関数のパラメータをガウス分布でフィッティング
# パラメータの最尤推定値からマナラノビス距離で等距離にあるパラメータを
# ランダムに選択してプロット
# pandasのscatter_plottingを使って三次元以上もプロットできる

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


def plotGaussian(mu, sigma):
    th = np.linspace(-np.pi, np.pi, 1000)
    r = chi2.ppf(0.7, 2) / (np.linalg.eigvals(sigma)[0] + np.linalg.eigvals(sigma)[1])
    # r = np.random.uniform(low = chi2.ppf(0.8, 2) / (np.linalg.eigvals(sigma)[0] + np.linalg.eigvals(sigma)[1]), high = chi2.ppf(0.9999, 2) / (np.linalg.eigvals(sigma)[0] + np.linalg.eigvals(sigma)[1]), size = 1000)
    x_plot, y_plot = r * np.cos(th), r * np.sin(th)
    x_plot *= np.linalg.eigvals(sigma)[0] ** 0.5
    y_plot *= np.linalg.eigvals(sigma)[1] ** 0.5
    X = np.dot(np.c_[x_plot, y_plot], np.linalg.eig(sigma)[1].T)
    X += mu
    for i in range(100):
        num = np.random.randint(1000)
        x = np.linspace(-2, 6, 1000)
        y = X[num, 0] + X[num, 1] * x
        plt.plot(x, y, c="r", alpha=0.25)
    plt.plot(x, mu[0] + mu[1] * x, c="r")
    plt.figure(0)
    x, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(x, y)

    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx;
    pos[:, :, 1] = yy
    rv = multivariate_normal(mu, sigma)
    # plt.pcolor(x, y, rv.pdf(pos))
    plt.contourf(x, y, rv.pdf(pos))
    plt.plot(X[:, 0], X[:, 1])
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    N = 100
    x = np.linspace(0, 5, N)
    y = np.random.normal(0.5 * x + 1, 1)
    phi = np.zeros([len(x), 2])
    phi[:, 0] = 1
    phi[:, 1] = x
    alpha, beta = 0.1, 0.01
    sigma = np.linalg.inv(alpha * np.eye(2) + beta * np.dot(phi.T, phi))

    mu = beta * np.dot(np.dot(sigma, phi.T), y)
    print
    mu
    plt.plot(x, y)
    plt.scatter(x, y)
    plotGaussian(mu, sigma)
