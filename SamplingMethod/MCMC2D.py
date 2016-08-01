# coding:utf-8
"""
一次元ガウス分布を生成する
"""
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return multigauss(x, [0, 0], np.array([[1, 0.5], [0.5, 1]]) * 10) + multigauss(x, [1.2, 1], 10 * np.array([[1, 0.5], [0.5, 1]]))
def gauss(x, mu, sigma):
    return 1 / (2 * np.pi * sigma) ** 0.5 * np.exp(-(x - mu) ** 2 / (2 * sigma))
def multigauss(x, mu, sigma):
    D = len(x)
    sigmaInv = np.linalg.inv(sigma)
    sigmaDet = np.linalg.det(sigma)
    c = 1 / ((2 * np.pi) ** D / 2.0 * sigmaDet ** 0.5)
    centerX = (x - mu)
    A = 0
    for i in range(D):
        for j in range(D):
            A += centerX[i] * sigma[i, j] * centerX[j]
    return c * np.exp(-0.5 * A)

z = np.random.uniform(-0.5, 0.5, size = 2)
tmp = []
for i in range(10 ** 3):
    # 提案分布からzをサンプリング
    z_proposal = z + np.random.uniform(-0.5, 0.5, size = 2)
    # 棄却率を計算
    A = np.min([1, func(z_proposal) / func(z) ])
    u = np.random.uniform(0, 1)
    if A > u:
        z = z_proposal
    tmp.append(z)

xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = []
for x in np.c_[np.ravel(xx), np.ravel(yy)]:
    z.append(func(x))
z = np.array(z)
plt.pcolor(xx, yy, z.reshape([100, 100]))
plt.scatter(np.array(tmp)[:, 0], np.array(tmp)[:, 1])
plt.plot(np.array(tmp)[:, 0], np.array(tmp)[:, 1])
plt.show()
