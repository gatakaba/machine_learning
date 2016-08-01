# coding:utf-8
# D={0,1,...,0,1}
# 平均値μの確率分布を推定する
# p(D|μ)=Π μ^xn *(1-μ)^1-xn
from scipy.stats import beta

import matplotlib.pyplot as plt
import numpy as np
# 対数尤度 Σ　(xn *log(mu/(1-mu)+log(1-mu))
def loglikelihood(mu, D):
    return np.sum(D) * np.log(mu / (1 - mu)) + np.log(1 - mu) * len(D)
# 観測データ作成
# np.random.seed(0)
N = 100
D = np.random.binomial(1, 0.6, N)
# D = [1, 1, 1, 0, 0]
mu1, mu2 = [0.5, 0.5]
prev = 0
mylist = []
for i in range(10 ** 5 * 5):
    mu2 = np.random.normal(mu1, 0.01)
    l = loglikelihood(mu2, D)
    if l > prev or np.exp(l - prev) > np.random.random():
        mu1 = mu2
    mylist.append(mu1)
    prev = l
mylist = np.array(mylist)
mylist = np.array(mylist)[len(mylist) / 2:]
print np.mean(D), np.var(D)
print np.mean(mylist, axis=0)

plt.subplot(121)
plt.plot(mylist)
plt.subplot(122)
plt.hist(mylist, bins=30, normed=True)
x = np.linspace(0, 1, 1000)
y = beta.pdf(x, sum(D), len(D) - sum(D))
plt.plot(x, y, linewidth=3)
plt.show()
