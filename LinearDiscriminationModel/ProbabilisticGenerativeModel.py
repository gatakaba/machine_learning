import numpy as np
import matplotlib.pyplot as plt

def twoClass():
    # 2class linear discriminant
    N = 1000
    m1 = np.array([-5, 0])
    m2 = np.array([2, -5])
    sigma = np.array([[0.5, -2], [-2, 0.5]])

    # class1 data
    x1 = np.random.multivariate_normal(m1, sigma, N)
    # class2 data
    x2 = np.random.multivariate_normal(m2, sigma, N)
    # combine
    X = np.r_[x1, x2]
    w = np.dot(np.linalg.inv(sigma), (m1 - m2))
    w0 = -0.5 * np.dot(m1, np.dot(np.linalg.inv(sigma), m1)) + 0.5 * np.dot(m2, np.dot(np.linalg.inv(sigma), m2))

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)
    X = np.c_[np.ravel(xx), np.ravel(yy)]
    p1 = 1 / (1 + np.exp(-np.dot(w[np.newaxis], X.T))).reshape(100, 100)

    plt.pcolor(x, y, p1)
    plt.colormaps()
    plt.scatter(x1[:, 0], x1[:, 1], c="red")
    plt.scatter(x2[:, 0], x2[:, 1])

    plt.show()
def threeClass():
    # 4class linear discriminant
    N = 100
    m1 = np.array([0, 0])
    m2 = np.array([4, 2])
    m3 = np.array([-4, 0])
    m4 = np.array([0, -4])

    # sigma must be varience covariance matrix
    sigma1 = np.array([[1.0, 0.0], [0.0, 10.0]]) * 0.1 * 2
    sigma2 = np.array([[10.0, 3.0], [3.0, 10.0]]) * 0.1
    sigma3 = np.array([[1.0, 2.0], [2.0, 10.0]]) * 0.1 
    sigma4 = np.array([[10.0, 0.0], [0.0, 10.0]]) * 0.1 
    # class data
    x1 = np.random.multivariate_normal(m1, sigma1, N)
    x2 = np.random.multivariate_normal(m2, sigma2, N)
    x3 = np.random.multivariate_normal(m3, sigma3, N)
    x4 = np.random.multivariate_normal(m4, sigma4, N)
    sigma1 = np.cov(x1.T)
    sigma2 = np.cov(x2.T)
    sigma3 = np.cov(x3.T)
    sigma4 = np.cov(x4.T)

    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    xx, yy = np.meshgrid(x, y)
    X = np.c_[np.ravel(xx), np.ravel(yy)]
    def calcweight(m, sigma):
        sigmaInv = np.linalg.inv(sigma)
        return [np.dot(sigmaInv, m), -0.5 * np.dot(m, np.dot(sigmaInv, m)) + np.log(1 / 4.0)]

    w1 , w10 = calcweight(m1, sigma1)
    a1 = np.dot(w1[np.newaxis], X.T) .reshape(300, 300) + w10

    w2 , w20 = calcweight(m2, sigma2)
    a2 = np.dot(w2[np.newaxis], X.T).reshape(300, 300) + w20

    w3 , w30 = calcweight(m3, sigma3)
    a3 = np.dot(w3[np.newaxis], X.T) .reshape(300, 300) + w30

    w4 , w40 = calcweight(m4, sigma4)
    a4 = np.dot(w4[np.newaxis], X.T) .reshape(300, 300) + w40

    p1 = np.exp(a1) / (np.exp(a1) + np.exp(a2) + np.exp(a3) + np.exp(a4))
    p2 = np.exp(a2) / (np.exp(a1) + np.exp(a2) + np.exp(a3) + np.exp(a4))
    p3 = np.exp(a3) / (np.exp(a1) + np.exp(a2) + np.exp(a3) + np.exp(a4))
    p4 = np.exp(a4) / (np.exp(a1) + np.exp(a2) + np.exp(a3) + np.exp(a4))

    plt.pcolor(x, y, p1)
    plt.colorbar()
    plt.scatter(x1[:, 0], x1[:, 1], c="red")
    plt.scatter(x2[:, 0], x2[:, 1], c="green")
    plt.scatter(x3[:, 0], x3[:, 1], c="blue")
    plt.scatter(x4[:, 0], x4[:, 1], c="yellow")
    plt.show()

    from mayavi import mlab
    mlab.surf(p1 * 100)
    mlab.surf(p2 * 100)
    mlab.surf(p3 * 100)
    mlab.surf(p4 * 100)
    mlab.show()

threeClass()


