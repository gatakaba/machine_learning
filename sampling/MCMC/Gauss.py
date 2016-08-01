# coding:utf-8
# ガウス分布のパラメータをMCMCによって推定する
import time

from matplotlib.ticker import LogLocator

import matplotlib.pyplot as plt
import numpy as np
# 対数尤度の計算
# L(mu,sigmma)=-np.dot((mu - D), (mu - D)) / (2 * sigma) - N / 2 *
# np.log(2 * np.pi * sigma)


def loglikelihood(D, mu, sigma):
    return -np.dot((mu - D), (mu - D)) / sigma - N * np.log(2 * np.pi * sigma)

# 観測データ作成
np.random.seed(0)
N = 100
D = np.random.normal(3, 3, size=N)
print D
mu1, mu2 = [0, 0]
sigma1, sigma2 = [1, 0]
prev = 0
mylist = []
a = time.time()
for i in range(10 ** 6):
    mu2 = np.random.normal(mu1, 1)
    l = loglikelihood(D, mu2, sigma2)
    # 尤度が高くなった場合移動
    if l > prev or np.exp(l - prev) > np.random.random():
        mu1 = mu2
    mylist.append([mu1, sigma1])

    sigma2 = np.random.normal(sigma1, 0.5)
    l = loglikelihood(D, mu2, sigma2)
    if l > prev or np.exp(l - prev) > np.random.random():
        sigma1 = sigma2
    mylist.append([mu1, sigma1])

    prev = l
print time.time() - a
# 以下描画処理
mylist = np.array(mylist)[len(mylist) / 2:]
print np.mean(D), np.var(D)
print np.mean(mylist, axis=0)

plt.subplot(121)
plt.plot(mylist)
plt.subplot(122)
plt.hist(mylist, bins=30, normed=True)
plt.show()
