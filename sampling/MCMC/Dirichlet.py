# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def loglikelihood(mu, D):
    return np.dot(np.sum(D, axis = 0), np.log(mu))

mu = [0.5, 0.3, 0.2]
K=len(mu)
N = 10
D = np.zeros([N, K])
d = np.random.choice([0, 1, 2], size = N, p = mu)
for i in range(3): 
    D[np.where(d == i)[0], i] = 1
print np.sum(D, axis = 0)
prev = 0
mylist = []
mu1 = np.ones(K) / K
for i in range(10 ** 5):
    mu2 = np.random.normal(mu1, 0.01)
    mu2 = mu2 / np.sum(mu2)
    l = loglikelihood(mu2, D)
    if l > prev or np.exp(l - prev) > np.random.random():
        mu1 = mu2
    mylist.append(mu1)
    prev = l
mylist = np.array(mylist)
mylist = np.array(mylist)[len(mylist) / 2:]

print np.mean(mylist, axis = 0)
plt.subplot(121)
plt.plot(mylist)
plt.subplot(122)
plt.hist(mylist, bins = 30, normed = True)
plt.show()