# coding:utf-8
"""
入力:N次元
出力:1次元
のパーセプトロン
閾値関数はシグモイド関数を使用
"""
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, N, c=1):
        # N:次元数
        # C:シグモイド係数
        self.N = N
        self.c = c
        self.w = np.random.normal(size=N + 1)

    def fit(self, X, t):
        """
        X:サンプルデータ
        t:教師データ
        """
        X = np.c_[np.ones(len(X)), X]
        for j in range(10 ** 4):
            E = 0
            for i in range(len(X)):
                u = np.dot(self.w, X[i])
                y = self.sigmoid(u)
                dE = (t[i] - y) * self.diffsigmoid(u) * X[i]
                self.w += dE * 10 ** 0
                E += np.abs((t[i] - y))
            print(E)

    def predict(self, x):
        u = np.dot(self.w, np.r_[1, x])
        y = self.sigmoid(u)
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x * self.c))

    def diffsigmoid(self, x):
        tmp = self.sigmoid(x)
        return tmp * (1 - tmp) * self.c


if __name__ == "__main__":
    # andを学習する
    np.random.seed(0)
    N = 2
    AndData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    AndTarget = np.array([0, 0, 0, 1])

    OrData = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    OrTarget = np.array([0, 1, 1, 1])

    X, t = AndData, AndTarget

    p = Perceptron(N)
    p.fit(X, t)

    for x in X:
        print
        x, p.predict(x)

    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    tmp = np.c_[np.ravel(xx), np.ravel(yy)]
    z = np.zeros(100 ** 2)
    for i in range(100 ** 2):
        z[i] = p.predict(tmp[i])
    plt.pcolor(xx, yy, z.reshape([100, 100]))
    plt.colorbar()
    plt.show()
