# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def twoClass():
    # 2class linear discriminant
    N = 100
    # class1 data
    x1 = np.random.normal(loc=[0, 0], size=[N, 2])
    # add bias
    x1 = np.c_[np.ones(N), x1]
    # target
    t1 = np.zeros([N, 2])
    t1[:, 0] = 1

    # class2 data
    x2 = np.random.normal(loc=[2, 4], size=[N, 2])
    x2 = np.c_[np.ones(N), x2]
    t2 = np.zeros([N, 2])
    t2[:, 1] = 1
    # combine
    X = np.r_[x1, x2]
    T = np.r_[t1, t2]
    # learn
    W = np.dot(np.linalg.pinv(X), T).T
    # plotting prepare
    x = np.linspace(-5, 5)
    y = -((W[0, 1] - W[1, 1]) * x + (W[0, 0] - W[1, 0])) / (W[0, 2] - W[1, 2])

    plt.plot(x, y)
    plt.scatter(x1[:, 1], x1[:, 2], c="red")
    plt.scatter(x2[:, 1], x2[:, 2])
    plt.show()

def threeClass():
    # 3class linear discriminant
    N = 1000
    M = 3
    # class1 data
    x1 = np.random.normal(loc=[0, 0], size=[N, 2])
    # add bias
    x1 = np.c_[np.ones(N), x1]
    # target
    t1 = np.zeros([N, M])
    t1[:, 0] = 1
    # class2 data
    x2 = np.random.normal(loc=[2, 10], size=[N, 2])
    x2 = np.c_[np.ones(N), x2]
    t2 = np.zeros([N, M])
    t2[:, 1] = 1
    # class2 data
    x3 = np.random.normal(loc=[-4, 4], size=[N, 2])
    x3 = np.c_[np.ones(N), x3]
    t3 = np.zeros([N, M])
    t3[:, 2] = 1

    # combine
    X = np.r_[x1, x2, x3]
    T = np.r_[t1, t2, t3]
    # learn
    W = np.dot(np.linalg.pinv(X), T).T

    # plotting prepare
    x = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
    plt.xlim(np.min(X[:, 1]), np.max(X[:, 1]))
    plt.ylim(np.min(X[:, 2]), np.max(X[:, 2]))
    for nums in [(0, 1), (1, 2), (2, 0)]:
        y = -((W[nums[0], 1] - W[nums[1], 1]) * x + (W[nums[0], 0] - W[nums[1], 0])) / (W[nums[0], 2] - W[nums[1], 2])
        plt.plot(x, y)

    plt.scatter(x1[:, 1], x1[:, 2], c="red")
    plt.scatter(x2[:, 1], x2[:, 2], c="green")
    plt.scatter(x3[:, 1], x2[:, 2], c="yellow")
    plt.show()
threeClass()
