import numpy as np
import matplotlib.pyplot as plt


def Kernel(x,y):
    # x,y scalar
    sigma = 0.1
    return np.exp(-np.abs(x - y) ** 2 / (2 * sigma))
    

N = 100
M = 100
l = 10 ** -2
x = np.linspace(-np.pi , np.pi, N)
y = np.sin(x)
t = y + np.random.normal(loc=0, scale=0.1, size=len(x))

scale = np.linspace(-np.pi , np.pi, M)
k = np.empty([N, M])

xx = np.tile(x, [N, 1])

K = Kernel(xx, xx.T)
a = np.dot(np.linalg.inv(K + l * np.eye(N)), t)

xx, ss = np.meshgrid(x, scale)
k = Kernel(xx, ss)

y_predict = np.dot(k.T, a)

plt.xlim([-np.pi, np.pi])
plt.plot(x, y, "ro-")
plt.plot(scale, y_predict, linewidth=3)
plt.plot(x, t, "o")
plt.show()
