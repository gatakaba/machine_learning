# coding:utf-8
"""
2014/02/09
Relavent Vector Machine Regression
"""
import matplotlib.pyplot as plt
import numpy as np


class RVM(object):

    def __init__(self, M=10):
        self.M = M
        self.alpha = np.abs(np.random.normal(size=self.M))
        self.beta = np.abs(np.random.normal(0, 1))

    def fit(self, X, y, verbose=False):
        phi = self.calc_phi(X, self.M)
        for i in range(1000):
            self.sigma = np.linalg.inv(
                np.diag(self.alpha) + self.beta * np.dot(phi.T, phi))

            self.m = self.beta * np.dot(np.dot(self.sigma, phi.T), y)

            gamma = 1 - self.alpha * np.diag(self.sigma)
            self.alpha = gamma / self.m ** 2
            beta_inv = np.linalg.norm(
                y - np.dot(phi, self.m)) ** 2 / (N - np.sum(gamma))
            self.beta = beta_inv ** -1
            if verbose:
                print self.log_likelihood(X, y)

    def predict(self, X, std=False):
        phi = self.calc_phi(X, M)
        mean = np.dot(phi, self.m)
        if not std:
            return mean
        else:
            N = len(X)
            std = np.zeros(N)
            for i in range(N):

                std[i] = self.beta ** -1 + \
                    np.dot(np.dot(phi[i].T, self.sigma), phi[i])
            return mean, std

    def score(self, X, y):
        y_predict = self.predict(X)
        u = (y - y_predict) ** 2
        v = (y - np.mean(y)) ** 2
        return 1 - u / v

    def log_likelihood(self, X, y):
        N = len(y)
        try:
            A_inv = np.linalg.inv(np.diag(self.alpha))
        except:
            A_inv = np.linalg.inv(
                np.diag(self.alpha) + np.eye(len(self.alpha)) ** 10 ** -9)
        phi = self.calc_phi(X, M)
        C = np.eye(N) / self.beta + np.dot(np.dot(phi, A_inv), phi.T)
        return -0.5 * (N * np.log(2 * np.pi)
                       + np.log(np.linalg.det(C))
                       + np.dot(np.dot(y, np.linalg.inv(C)), y))

    def calc_phi(self, x, M):
        sigma = 0.01
        return np.array([np.exp(-(x - i) ** 2 / sigma) for i in np.linspace(0, 1, M)]).T
        # return np.array([1 / (1 + np.exp(-sigma * (x - i))) for i in
        # np.linspace(0, 1, M)]).T

if __name__ == "__main__":
    # np.random.seed(0)
    M = 10
    N = 10
    # generate sample
    x_sample = np.random.uniform(0, 1, N)
    t_sample = np.sin(2 * np.pi * 3 * x_sample) + \
        np.random.normal(0, 0.1, size=N)
    # fitting
    rvm = RVM(M)
    rvm.fit(x_sample, t_sample, verbose=False)

    x_draw = np.linspace(0, 1, 1000)

    mean, std = rvm.predict(x_draw, std=True)

    plt.fill_between(x_draw, mean - std, mean + std, alpha=0.5)
    plt.plot(x_draw, mean, "k", linewidth=3)
    plt.scatter(x_sample, t_sample)

    plt.show()
