# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def LinearModel(x, y):
    """
        model:y=w・x
        w=(tr(X)・X)^-1・tr(X)・y
    """
    X = x[np.newaxis].T
    Y = y[np.newaxis].T
    tmp1 = np.linalg.inv(np.dot(X.T, X))
    w = np.dot(np.dot(tmp1, X.T), Y)

def LinearRegression(x, y):

    phi = np.exp
    print phi(x)
    X = x[np.newaxis].T
    Y = y[np.newaxis].T

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.normal(size=10)
    y = 2 * x
    LinearRegression(x, y)
