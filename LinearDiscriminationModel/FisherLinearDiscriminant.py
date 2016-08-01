import numpy as np
import matplotlib.pyplot as plt


# 2class linear discriminant
N = 1000
# class1 data
x1 = np.random.normal(loc=[0, 0], size=[N, 2])
# target
t1 = np.zeros([N, 2])
t1[:, 0] = 1

# class2 data
x2 = np.random.normal(loc=[2, 4], size=[N, 2])
t2 = np.zeros([N, 2])
t2[:, 1] = 1
# combine
X = np.r_[x1, x2]
T = np.r_[t1, t2]

m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)
# within class covariance matrix
Sw = np.dot((x1 - m1).T, (x1 - m1)) + np.dot((x2 - m2).T, (x2 - m2))
w = np.dot(np.linalg.inv(Sw), (m2 - m1))
w = w / np.dot(w, w) ** 0.5
w0 = np.dot(w, np.mean(X, axis=0))

x = np.linspace(-5, 5)
y = -w[0] / w[1] * x + w0 / w[1]

plt.plot(x, y)
plt.scatter(x1[:, 0], x1[:, 1], c="red")
plt.scatter(x2[:, 0], x2[:, 1])
plt.show()
