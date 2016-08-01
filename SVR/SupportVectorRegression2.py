# coding:utf-8

import sys
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from PatternCoding import PatternCoding
import matplotlib.pyplot as plt
import numpy as np
pc = None


def func(X):
    t = []
    for i in X:
        x, y = i[0], i[1]
        if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.04:
            t.append(1)
        else:
            t.append(
                (1 + x) / 2 * np.sin(6 * np.pi * x ** 0.5 * y ** 2) ** 2)
    return np.array(t)


def linear(x1, x2):
    return np.dot(x1, x2)


def rbf(x1, x2):
    return np.exp(-np.sum((x1 - x2) ** 2) / 0.01)


def pt(x1, x2):
    phi1 = np.r_[pc.get_pattern(x1[0]), pc.get_pattern(x1[1])]
    phi2 = np.r_[pc.get_pattern(x2[0]), pc.get_pattern(x2[1])]
    return np.dot(phi1, phi2)


def sd(x1, x2):
    t1 = (pc.get_pattern(x1[0]) + 2) * pc.get_pattern(x1[1])
    t2 = (pc.get_pattern(x1[1]) + 2) * pc.get_pattern(x1[0])

    t3 = (pc.get_pattern(x2[0]) + 2) * pc.get_pattern(x2[1])
    t4 = (pc.get_pattern(x2[1]) + 2) * pc.get_pattern(x2[0])

    # phi1 = np.r_[t1, t2]
    # phi2 = np.r_[t3, t4]
    # return np.dot(phi1, phi2)

    return (np.sum(t1 * t3) + np.sum(t2 * t4)) / 4


def kernel(x1, x2):

    func = sd

    N = len(x1)
    M = len(x2)
    K = np.empty([N, M])

    if N == M:
        for i in range(N):
            for j in range(i, M):
                K[i, j] = func(x1[i], x2[j])
                K[j, i] = K[i, j]
    else:
        for i in range(N):
            for j in range(M):
                K[i, j] = func(x1[i], x2[j])
    return K

if __name__ == "__main__":

    if len(sys.argv) > 1:
        q = int(sys.argv[1])
        n = int(sys.argv[2])
        r = int(sys.argv[3])
        N = int(sys.argv[4])
        C = float(sys.argv[5])
    else:
        q = 10000
        n = 10000
        r = 3
        N = 2 ** 9
        C = 10 ** 9

    pc = PatternCoding(q, n, r)
    X = np.random.uniform(0, 1, size=[N, 2])

    y = func(X)
    t = y
    svr = SVR(kernel=kernel, C=C, epsilon=10 ** -1, tol=10 ** -3)
    svr.fit(X, t)
    print "finish learning"
    # 評価
    M = 100
    xx, yy = np.meshgrid(
        np.linspace(0, 1, M, endpoint=False), np.linspace(0, 1, M, endpoint=False))
    X_draw = np.c_[np.ravel(xx), np.ravel(yy)]
    t_draw = func(X_draw).reshape([M, M])

    t_predict = svr.predict(X_draw).reshape([M, M])

    print np.mean((t_draw - t_predict) ** 2) ** 0.5

    # 以下描画処理
    plt.figure(0)
    plt.title("SVR target func & sample")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.pcolor(xx, yy, t_draw, vmin=0, vmax=1)
    plt.colorbar()
    """
    for i in svr.support_:
        plt.scatter(X[i, 0], X[i, 1], s=80, c='c', marker='o')
    """
    plt.scatter(X[:, 0], X[:, 1])

    plt.figure(1)
    plt.title("SVR predict")
    plt.pcolor(xx, yy, t_predict, vmin=0, vmax=1)
    plt.colorbar()

    plt.show()
