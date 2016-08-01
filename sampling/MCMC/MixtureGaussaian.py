# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np


def loglikelihood(x):
    a, b = 2.5, 2.5
    return -((x + a) ** 2 * (x - b) ** 2) / 4

x1 = 0
prev = 0
mylist = []
for i in range(10 ** 7):
    x2 = np.random.normal(x1, 1)
    l = loglikelihood(x2)
    if l > prev or np.exp(l - prev) > np.random.random():
        x1 = x2
    mylist.append(x1)

x_plot = np.linspace(-5, 5, 1000)
y_plot = np.exp(loglikelihood(x_plot))
y_plot /= np.trapz(y_plot, x_plot)

plt.subplot(211)
plt.plot(mylist)
plt.subplot(212)
plt.hist(mylist, bins=100, normed=True)
plt.plot(x_plot, y_plot)

plt.show()
