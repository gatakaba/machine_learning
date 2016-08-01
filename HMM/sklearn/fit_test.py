# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import hmm

model = hmm.GaussianHMM(3)
N = 100
x1 = np.random.normal([1, 1], size = [N, 2])
x2 = np.random.normal([-1, -1], size = [N, 2])
X = np.r_[x1, x2]

import random
random.shuffle(X)
model.fit([np.c_[X]])
t1 = np.random.normal([1 * 0, 1 * 0], size = [N, 2])
t2 = np.random.normal([-1 * 0, -1 * 0], size = [N, 2])

T = np.r_[t1, t2]
color = model.predict(np.c_[T])
plt.figure(0)
plt.scatter(T[:, 0], T[:, 1], c = color, alpha = 0.5)

plt.figure(1)
plt.plot(color)
plt.show()
