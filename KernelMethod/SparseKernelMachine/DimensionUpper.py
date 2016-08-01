# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
x1 = np.random.normal(-1, 0.1, size=100)
x2 = np.random.normal(0, 0.1, size=100)
x3 = np.random.normal(1, 0.1, size=100)

x = np.r_[x1, x2, x3]
t = ["r" if 100 < i < 200 else "b" for i in range(300)]


plt.scatter(x, np.zeros(300), c=t)
plt.scatter(x, x ** 2, c=t)

plt.xlim = [-3, 3]
plt.ylim = [-1, 1]
plt.show()
