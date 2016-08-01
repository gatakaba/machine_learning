# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

c1, c2 = [], []

for x in np.random.uniform(size = [1000, 2]):
    if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 > 0.05:
        c1.append(x)
    else:
        c2.append(x)
c1 = np.array(c1)
c2 = np.array(c2)
X = np.r_[c1, c2]
y1, y2, y3 = [], [], []
mu1 = [0.5, 0.5]
mu2 = [0.0, 0.0]

for x in X:
    y1.append(np.exp(1 * np.abs(x - mu1)))
    y2.append(np.exp(1 * np.abs(x - mu2)))

Y = np.r_[y1, y2]
plt.figure(0)
for i in range(len(c1)):
    plt.scatter(c1[i, 0], c1[i, 1], c = "b")
for i in range(len(c2)):
    plt.scatter(c2[i, 0], c2[i, 1], c = "r")
plt.figure(1)
for i in range(len(c1)):
    plt.scatter(Y[i, 0], Y[i, 1], c = "b")
for i in range(len(c1), len(c1) + len(c2)):
    plt.scatter(Y[i, 0], Y[i, 1], c = "r")
plt.show()
"""
plt.figure(0)
plt.scatter(X[:100, 0], X[:100, 1], c = "r")
plt.scatter(X[100:200, 0], X[100:200, 1], c = "b")
plt.scatter(X[200:300, 0], X[200:300, 1], c = "y")
plt.scatter(mu1[0], mu1[1], marker = "*", s = 48, c = "b")
plt.scatter(mu2[0], mu2[1], marker = "*", s = 48, c = "r")

plt.figure(1)
plt.scatter(Y[:100, 0], Y[:100, 1], c = "r")
plt.scatter(Y[100:200, 0], Y[100:200, 1], c = "b")
plt.scatter(Y[200:300, 0], Y[200:300, 1], c = "y")
plt.show()
"""