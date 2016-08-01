# coding:utf-8
# 一次元ガウス混合分布

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


def draw(mean, cov, pi):
    def gaussian(x, mean, cov):
        """多変量ガウス関数"""
        temp1 = 1 / ((2 * np.pi) ** (x.size / 2.0))
        temp2 = 1 / (np.linalg.det(cov) ** 0.5)
        temp3 = -0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean)
        return temp1 * temp2 * np.exp(temp3)

    N = 100
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, N), np.linspace(-2.5, 2.5, N))
    Xplot = np.c_[np.ravel(xx), np.ravel(yy)]
    z1, z2 = np.empty(N ** 2), np.empty(N ** 2)
    
    for i in range(N ** 2):
        z1[i] = gaussian(Xplot[i], mean[0], cov[0])
        z2[i] = gaussian(Xplot[i], mean[1], cov[1])

    
    # 等高線描画
    plt.contour(xx, yy, z1.reshape((N, N)))
    plt.contour(xx, yy, z2.reshape((N, N)))
    
    # 平均値描画
    plt.scatter(*mean[0], c = "r", marker = "*", s = 200)
    plt.scatter(*mean[1], c = "r", marker = "*", s = 200)
    plt.show()


N = 500
x1 = np.random.multivariate_normal([-2.5, 0], [[1, 0], [0, 1]], size = N)
x2 = np.random.multivariate_normal([0, -2.5], [[1, 0], [0, 1]], size = N)
x3 = np.random.multivariate_normal([2.5, 0], [[1, 0], [0, 1]], size = N)
x4 = np.random.multivariate_normal([0, 2.5], [[1, 0], [0, 1]], size = N)
X = np.r_[x1, x2, x3, x4]
# x1 = np.random.normal(-1, 0.5, N / 2)
# x2 = np.random.normal(1, 0.5, N / 2)
tmp, tmp2 = [], [] 
for i in range(2, 10):
    g = mixture.GMM(n_components = i, covariance_type = "full")
    g.fit(X)
    tmp.append(g.aic(X))
    tmp2.append(g.bic(X))
plt.figure(0)
plt.plot(range(2, 10), np.array(tmp), label = "AIC")
plt.plot(range(2, 10), np.array(tmp2), label = "BIC")
plt.legend()
plt.figure(1)
g = mixture.GMM(n_components = 2, covariance_type = "full")
g.fit(X)
plt.scatter(X[:, 0], X[:, 1])
draw(g.means_, g.covars_, g.weights_)

# x = np.linspace(-5, 5, 1000)
# y = g.predict_proba(x)

# plt.plot(x, y)
# plt.show()
