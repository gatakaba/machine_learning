# coding:utf-8
'''
2015/02/12　更新
PRML 下 P54参照
'''

from cvxopt import matrix, solvers

from PatternCoding import PatternCoding
import matplotlib.pyplot as plt
import numpy as np


class SVR(object):

    def __init__(self, C=10 ** 1, sigma=10 ** -2, epsilon=10 ** -1):
        self.C = C
        self.sigma = sigma
        self.epsilon = epsilon
        self.pc = PatternCoding(100, 100, 1)

    def kernel(self, x1, x2):
        # return np.dot(x1, x2)
        # return (1 + np.dot(x1, x2)) ** 3

        # return np.dot(np.exp(-x1 ** 2), np.exp(-x2 ** 2) * 1000.0)

        #phi1 = np.r_[self.pc.get_pattern(x1[0]), self.pc.get_pattern(x1[1])]
        #phi2 = np.r_[self.pc.get_pattern(x2[0]), self.pc.get_pattern(x2[1])]

        t1 = (self.pc.get_pattern(x1[0]) + 1) * self.pc.get_pattern(x1[1]) / 2
        t2 = (self.pc.get_pattern(x1[1]) + 1) * self.pc.get_pattern(x1[0]) / 2

        t3 = (self.pc.get_pattern(x2[0]) + 1) * self.pc.get_pattern(x2[1]) / 2
        t4 = (self.pc.get_pattern(x2[1]) + 1) * self.pc.get_pattern(x2[0]) / 2

        phi1 = np.r_[t1, t2]
        phi2 = np.r_[t3, t4]

        return np.dot(phi1, phi2)

        return np.exp(-np.sum((x1 - x2) ** 2) / self.sigma)

    def L(self, x, t):
        # 評価関数
        # 動作未確認
        pass
        """
        l = 0
        for n in range(N):
            for m in range(N):
                l += (self.a1[n] - self.a2[m]) * \
                    (self.a1[m] - self.a2[n]) * self.kernel(x[n], x[m])
        l = -0.5 * l - self.epsilon * \
            np.sum(self.a1 + self.a2) + np.sum((self.a1 - self.a2) * t)
        """

    def solve_qp(self, K, t):
        # solve quadratic programs
        # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # minimize_{x} (1/2) xPx+qx
        # subject to
        # Gx<h
        # Ax=b

        N = len(K)
        P = np.zeros([2 * N, 2 * N])
        P[0:N, 0:N] = K
        P = matrix(P)
        q = matrix(np.r_[-t, np.ones(N) * self.epsilon])

        G = np.zeros([4 * N, 2 * N])
        G[0 * N:1 * N, 0 * N:1 * N] = np.diag(np.ones(N))
        G[1 * N:2 * N, 0 * N:1 * N] = -np.diag(np.ones(N))
        G[2 * N:3 * N, 0 * N:1 * N] = -np.diag(np.ones(N))
        G[3 * N:4 * N, 0 * N:1 * N] = np.diag(np.ones(N))

        G[0 * N:1 * N, 1 * N:2 * N] = np.diag(np.ones(N))
        G[1 * N:2 * N, 1 * N:2 * N] = -np.diag(np.ones(N))
        G[2 * N:3 * N, 1 * N:2 * N] = np.diag(np.ones(N))
        G[3 * N:4 * N, 1 * N:2 * N] = - np.diag(np.ones(N))
        G = matrix(G)

        h = np.zeros(4 * N)
        h[0 * N:1 * N] = 2 * self.C
        h[1 * N:2 * N] = 0
        h[2 * N:3 * N] = 2 * self.C
        h[3 * N:4 * N] = 0
        h = matrix(h)

        A = matrix(np.r_[np.ones(N), np.zeros(N)], (1, 2 * N))
        b = matrix(0.0)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x']).reshape(2 * N)

    def fit(self, X, t):
        self.train_X = X
        N = len(X)
        K = np.empty([N, N])
        for i in range(N):
            for j in range(N):
                K[i, j] = self.kernel(self.train_X[i], self.train_X[j])
        u = self.solve_qp(K, t)
        b1 = u[:N]
        b2 = u[N:]
        self.a1 = (b1 + b2) / 2
        self.a2 = (b2 - b1) / 2
        self.SV_index = np.where(b2 > 10 ** -3)[0]

        tmp1 = 0
        for n in range(N):
            tmp2 = 0
            for m in self.SV_index:
                tmp2 += b1[m] * self.kernel(X[n], X[m])
            tmp1 += (t[n] - self.epsilon - tmp2)
        self.b = tmp1 / N

    def predict(self, X):
        N = len(X)
        y = np.empty(N)
        for i in range(N):
            for n in self.SV_index:
                y[i] += (self.a1[n] - self.a2[n]) * \
                    self.kernel(self.train_X[n], X[i])
        return y - self.b

if __name__ == "__main__":
    def func(X):
        return -np.sin(2 * np.pi * X[:, 0] + X[:, 1] ** 0.5) + X[:, 1] ** 2

    def func2(X):
        t = []
        for i in X:
            x, y = i[0], i[1]
            if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.04:
                t.append(1)
            else:
                t.append(
                    (1 + x) / 2 * np.sin(6 * np.pi * x ** 0.5 * y ** 2) ** 2)
        return np.array(t)
    N = 500
    X = np.random.uniform(0, 1, size=[N, 2])

    y = func2(X)
    t = y
    svr = SVR()
    svr.fit(X, t)
    # 以下描画処理

    M = 33
    xx, yy = np.meshgrid(np.linspace(0, 0.99, M), np.linspace(0, 0.99, M))
    X_draw = np.c_[np.ravel(xx), np.ravel(yy)]
    t_draw = func2(X_draw).reshape([M, M])
    t_predict = svr.predict(X_draw).reshape([M, M])

    plt.figure(0)
    plt.title("SVR target func & sample")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.pcolor(xx, yy, t_draw, vmin=0, vmax=1)
    plt.colorbar()
    for i in svr.SV_index:
        plt.scatter(X[i, 0], X[i, 1], s=80, c='c', marker='o')
    plt.scatter(X[:, 0], X[:, 1])
    plt.savefig("svr_sample.png")

    plt.figure(1)
    plt.title("SVR predict")
    plt.pcolor(xx, yy, t_predict, vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("svr_predict.png")
    plt.show()
