# coding:utf-8
"""
線形SVM
"""
from cvxopt import matrix, solvers

import matplotlib.pyplot as plt
import numpy as np


def qp(K, C=10 ** 0):
    P = matrix(K)
    q = matrix(-np.ones(N))
    temp1 = np.diag([-1.0] * N)
    temp2 = np.identity(N)
    G = matrix(np.vstack((temp1, temp2)))
    h = matrix(np.zeros(N))
    temp1 = np.zeros(N)
    temp2 = np.ones(N) * C
    h = matrix(np.hstack((temp1, temp2)))
    A = matrix(t, (1, N))
    b = matrix(0.0)
    sol = solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x']).reshape(N)
    return a


def linear(x1, x2):
    return np.dot(x1, x2)


def gauss(x1, x2, sigma=1):
    return np.exp(-np.sum(np.abs(x1 - x2)) / sigma)


def poly(x1, x2, p=2):
    return (1 + np.dot(x1, x2)) ** p


def f(x, a, b, S):
    tmp = 0
    for n in S:
        tmp += a[n] * t[n] * kernel(x, X[n])
    return tmp + b
if __name__ == "__main__":
    # サンプル作成
    N = 300
    x1 = np.random.multivariate_normal(
        [-1, 1], np.array([[0.1, 0], [0, 0.1]]), size=N / 3)
    x2 = np.random.multivariate_normal(
        [0, 1], np.array([[0.1, 0], [0, 0.1]]), size=N / 3)
    x3 = np.random.multivariate_normal(
        [1, 1], np.array([[0.1, 0], [0, 0.1]]), size=N / 3)
    t1, t2, t3 = np.ones(N / 3), -np.ones(N / 3), np.ones(N / 3)

    X = np.r_[x1, x2, x3]
    t = np.r_[t1, t2, t3]
    kernel = gauss
    # カーネル行列作成
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            K[i, j] = t[i] * t[j] * kernel(X[i], X[j])
    # ラグランジュ定数の算出
    a = qp(K)
    # サポートベクトルのインデックスを抽出
    S = np.where(a > 0.001)[0]
    print a[S]
    # wを計算
    w = np.zeros(2)
    for n in S:
        w += a[n] * t[n] * X[n]
     # bを計算
    sum = 0
    for n in S:
        temp = 0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        sum += (t[n] - temp)
    b = sum / len(S)
    print S, b
    # 訓練データを描画
    plt.plot(x1[:, 0], x1[:, 1], 'bx')
    plt.plot(x2[:, 0], x2[:, 1], 'rx')
    plt.plot(x3[:, 0], x3[:, 1], 'bx')
    # サポートベクトルを描画
    for n in S:
        plt.scatter(X[n, 0], X[n, 1], s=80, c='c', marker='o')
    # 識別境界を描画
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    z = []
    for i in np.c_[np.ravel(xx), np.ravel(yy)]:
        z.append(f(i, a, b, S))
    zz = np.array(z).reshape([100, 100])
    # plt.xlim([np.min(X[:, 0]) * 1.5, np.max(X[:, 0]) * 1.5])
    # plt.ylim([np.min(X[:, 1]\) * 1.5, np.max(X[:, 1]) * 1.5])
    # plt.pcolor(xx, yy, np.sign(zz))
    plt.imshow(np.sign(zz), interpolation="nearest",
               origin="lower", extent=[-2, 2, -2, 2])
    plt.contour(xx, yy, zz, [0.0], colors='k', linewidths=1, origin='lower')
    plt.show()
