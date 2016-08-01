# coding:utf-8
"""
平均と分散が不明な場合における分布推定
"""
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


def log_likelihood(x, mu, sigma):
    n = len(x)
    return -n / 2 * np.log(2 * np.pi * sigma ** 2) - np.sum((x - mu) ** 2) / (2 * sigma ** 2)
x_sample = norm(5, 2).rvs(size=1000)

mu1 = 0
sigma1 = 0.1

prev = 0
mylist = []
for i in range(300000):
    mu2 = np.random.normal(mu1, 0.1)
    sigma2 = np.random.normal(sigma1, 0.1)
    l = log_likelihood(x_sample, mu2, sigma2)

    if l > prev or np.exp(l - prev) > np.random.random():
        mu1 = mu2
        sigma1 = sigma2

    mylist.append([mu1, sigma1])
    prev = l

mylist = np.array(mylist)
mu = mylist[:, 0]
sigma = mylist[:, 1]
print np.mean(mu), np.std(mu)
print np.mean(sigma), np.std(sigma)
plt.figure(0)
plt.suptitle("sampling")
plt.subplot(211)
plt.title("mean")
plt.plot(mu[100000:])
plt.subplot(212)
plt.title("variance")
plt.plot(np.abs(sigma[100000:]))
plt.savefig("sampling")

plt.figure(1)
plt.suptitle("Posterior distribution")
plt.subplot(211)
plt.title("mean")
plt.hist(mu[100000:], bins=100, normed=True)
plt.subplot(212)
plt.title("variance")
plt.hist(np.abs(sigma[100000:]), bins=100, normed=True)
plt.savefig("posterior_distribution")
plt.show()
