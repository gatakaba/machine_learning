import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

def predict1D():
    X = np.random.normal(0, 1, size = 1000)[:, np.newaxis]
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.1).fit(X)
    
    x = np.linspace(-2, 2, 1000)[:, np.newaxis]
    y = kde.score_samples(x)
    plt.hist(X, bins = 20, normed = True, alpha = 0.5)
    plt.plot(x, np.exp(y))
    plt.show()

def predict2D():
    
    X = np.random.multivariate_normal([1, 1], [[2, 1], [1, 1]], size = 1000)
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.5).fit(X)
    
    x, y = np.linspace(-2, 2, 300), np.linspace(-2, 2, 300)
    xx, yy = np.meshgrid(x, y)
    zz = kde.score_samples(np.c_[np.ravel(xx), np.ravel(yy) ]).reshape([300, 300])
    
    plt.pcolor(xx, yy, np.exp(zz))
    # plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    
if __name__=="__main__":
    predict2D()