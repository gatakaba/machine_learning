# coding:utf-8
"""
カーネル行列を作成
サンプル数をN
基底関数の数をMとすると、
N×M行列になる
"""
import numpy as np


def gauss(x, M, s=0.01):
    m = np.linspace(-0.5, 0.5, M)
    mm, xx = np.meshgrid(m, x)
    return np.exp(-(xx - mm) ** 2 / (2 * s ** 2))


def sigmoid(x, M):
    m = np.linspace(-0.5, 0.5, M)
    mm, xx = np.meshgrid(m, x)
    s = 100
    return 1 / (1 + np.exp(-s * (xx - mm)))


def poly(x, M):
    m = range(0, M)
    mm, xx = np.meshgrid(m, x)
    return xx ** mm


def random(x, M):
    return np.random.uniform(-0.1, 0.1, size=[len(x), M])


def digit(x, M):
    return np.sign(np.random.uniform(-0.1, 0.1, size=[len(x), M]))


def gauss_sigmoid(x, M):
    return np.r_[gauss(x, int(M / 2)), sigmoid(x, int(M / 2))]


if __name__ == "__main__":
    N = 100
    M = 10

    x = np.linspace(-0.5, 0.5, N)
    print(gauss(x, M).shape)
