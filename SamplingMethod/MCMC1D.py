# coding:utf-8
"""
一次元ガウス分布を生成する
"""
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return gauss(x, 0, 1)
def gauss(x, myu, sigma):
    return 1 / (2 * np.pi * sigma) ** 0.5 * np.exp(-(x - myu) ** 2 / (2 * sigma))
    
z = 0
tmp = []
for i in range(10 ** 4):
    # 提案分布からzをサンプリング
    z_proposal = z + np.random.uniform(-0.5, 0.5)
    # 棄却率を計算
    A = np.min([1, func(z_proposal) / func(z) ])
    u = np.random.uniform(0, 1)
    if A > u:
        z = z_proposal
    tmp.append(z)
plt.hist(tmp, bins = 100, normed = True)
x = np.linspace(-5, 5, 1000)
c = np.trapz(func(x), x)
plt.plot(x, func(x) / c, linewidth = 3)
plt.show()