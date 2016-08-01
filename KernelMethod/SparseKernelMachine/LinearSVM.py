# coding:utf-8
"""
2015/02/10 更新
線形ハードマージンSVM
PRML 下 P38 参照
"""
from cvxopt import matrix, solvers

import matplotlib.pyplot as plt
import numpy as np


class LinearSVM(object):

    def __init__(self):
        pass

    def solve_qp(self, K):
        # solve quadratic programs
        # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # minimize_{x} (1/2) xPx+qx (1)
        # subject to
        # Gx<h (2)
        # Ax=b (3)
        # (1): Lagrange func L(a)
        # (2): a_{n} >0 ,
        # (3): \sum(a_{n} t_{n}) =0
        N = len(K)
        P = matrix(K)
        q = matrix(-np.ones(N))
        G = matrix(np.diag([-1.0] * N))
        h = matrix(np.zeros(N))
        A = matrix(t, (1, N))
        b = matrix(0.0)
        sol = solvers.qp(P, q, G, h, A, b)
        a = np.array(sol['x']).reshape(N)
        return a

    def kernel(self, x1, x2):
        return np.dot(x1, x2)

    def fit(self, X, y):
        self.train_X = X
        K = np.zeros([N, N])
        # カーネル行列作成
        for i in range(N):
            for j in range(i, N):
                K[i, j] = t[i] * y[j] * self.kernel(X[i], X[j])
        K = K.T

        # ラグランジュ定数の算出
        self.a = self.solve_qp(K)
        self.at = self.a * y
        self.SV_index = np.where(self.a > 0.001)[0]
        # 切片bの導出
        tmp1 = 0
        for n in self.SV_index:
            tmp2 = 0
            for m in self.SV_index:
                tmp2 += self.at[m] * self.kernel(X[n], X[m])
            tmp1 += (y[n] - tmp2)
        self.b = tmp1 / len(self.SV_index)

    def predict(self, x):
        y = 0
        for n in self.SV_index:
            y += self.at[n] * self.kernel(self.train_X[n], x)
        return y + self.b

if __name__ == "__main__":
    # サンプル作成
    N = 100
    mu1 = np.array([-1, 1]) / 2
    mu2 = np.array([1, -1])
    sigma = np.array([[0.1, 0], [0, 0.1]])
    x1 = np.random.multivariate_normal(mu1, sigma, size=int(N / 2.0))
    x2 = np.random.multivariate_normal(mu2, sigma, size=int(N / 2.0))
    t1, t2 = np.ones(N / 2), -np.ones(N / 2)
    X = np.r_[x1, x2]
    t = np.r_[t1, t2]
    linear_svm = LinearSVM()

    linear_svm.fit(X, t)

    # 識別境界を描画
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    z = []
    for i in np.c_[np.ravel(xx), np.ravel(yy)]:
        z.append(linear_svm.predict(i))
    zz = np.array(z).reshape([100, 100])
    # plt.xlim([np.min(X[:, 0]) * 1.5, np.max(X[:, 0]) * 1.5])
    # plt.ylim([np.min(X[:, 1]\) * 1.5, np.max(X[:, 1]) * 1.5])
    # plt.pcolor(xx, yy, np.sign(zz))
    plt.imshow(np.sign(zz), interpolation="nearest",
               origin="lower", extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
    plt.imshow(zz, interpolation="nearest",
               origin="lower", extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
    plt.colorbar()
    plt.contour(xx, yy, zz, [0.0], colors='k', linewidths=1, origin='lower')

    # サポートベクタの描画
    for i in linear_svm.SV_index:
        if t[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], s=80, c='c', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=80, c='c', marker='o')
    # データサンプルの描画
    for i in range(N):
        if t[i] == -1:
            plt.scatter(X[i, 0], X[i, 1], c="r")
        else:
            plt.scatter(X[i, 0], X[i, 1], c="b")
    plt.show()
