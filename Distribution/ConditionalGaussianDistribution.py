# coding:utf-8
#2つ以上の変数がガウス分布に従うなら,
# 一方の変数が与えられた場合、もう一方の確率分布もガウス分布になります。
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
mu = [0.5, 0.3, 0.2,0.2]
sigma = np.eye(4) + np.random.normal(0, 0.1, size = [4, 4])
x = np.array([0.2, -0.2, -0.2])

mu1 = mu[0] - sigma[0, 0] ** -1 * np.dot(x - mu[1:], sigma[0, 1:])

x = np.linspace(-3, 3, 1000)
y = np.exp(-(x - mu1) ** 2)
print mu1
plt.plot(x, y)
plt.show()








