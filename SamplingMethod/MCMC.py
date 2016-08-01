# coding:utf-8
"""
平均(0,0)
共分散行列
[[1,0.7],
[0.7,1.0]]
に従うガウス分布を生成する
"""
import numpy as np
import matplotlib.pyplot as plt

x = -10
y = -10

tmp = []
for i in range(1000):
    tmp.append([x, y])
    x = np.random.normal(0.7 * y, 1 - 0.7 ** 2)
    y = np.random.normal(0.7 * x, 1 - 0.7 ** 2)
tmp = np.array(tmp)
plt.scatter(tmp[:, 0], tmp[:, 1])
plt.plot(tmp[:, 0], tmp[:, 1])
plt.show()
    