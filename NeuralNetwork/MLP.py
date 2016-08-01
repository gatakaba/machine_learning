# coding:utf-8
from blz.tests.common import verbose
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

import matplotlib.pyplot as plt
import numpy as np


class MLP(object):
    # pybrain wrapper

    def __init__(self, n_hidden_unit=3, n_iter=100, error=10 ** -3, verbose=False):
        if type(n_hidden_unit) is int:
            self.n_hidden_unit = [n_hidden_unit]
        elif type(n_hidden_unit) is list:
            self.n_hidden_unit = n_hidden_unit
        self.n_iter = n_iter
        self.error = error
        self.verbose = verbose

    def fit(self, x, y):
        if not len(x) == len(y):
            raise ValueError("The size of the array is different")
        if len(x.shape) == 1:
            self.n_input_unit = 1
        else:
            self.n_input_unit = x.shape[1]
        if len(y.shape) == 1:
            self.n_output_unit = 1
        else:
            self.n_output_unit = y.shape[1]

        self.layer = [self.n_input_unit] + \
            self.n_hidden_unit + [self.n_output_unit]
        self.net = buildNetwork(*self.layer)

        ds = SupervisedDataSet(self.n_input_unit, self.n_output_unit)
        for i in range(len(x)):
            ds.addSample(x[i], y[i])
        trainer = BackpropTrainer(self.net, ds)

        error = trainer.train()
        cnt = 1
        while cnt < self.n_iter and error > self.error:
            error = trainer.train()
            cnt += 1
            if self.verbose and cnt % 10 == 0:
                print cnt, error

    def predict(self, x):
        y = []
        if self.n_input_unit == 1:
            for i in x:
                y.append(self.net.activate([i]))
        else:
            for i in x:
                y.append(self.net.activate(i))
        return np.array(y)

    def score(self, x, y):
        u = np.sum((y - self.predict(x)) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v

if __name__ == "__main__":
    # sample code
    # four layer perceptron
    def f(X):
        return np.exp(-((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) * 10)

    np.random.seed(0)
    mlp = MLP([100, 10], n_iter=1000, error=10 ** -3, verbose=True)

    N = 1000
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(x, y)
    X = np.c_[np.ravel(xx), np.ravel(yy)]
    np.random.shuffle(X)

    train_X = X[:N]
    train_t = f(train_X)
    test_X = np.c_[np.ravel(xx), np.ravel(yy)]
    test_t = f(test_X)

    mlp.fit(train_X, train_t)
    y_predict = mlp.predict(test_X)
    plt.title("target & sample")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.pcolor(xx, yy, test_t.reshape([101, 101]), vmin=0, vmax=1)
    plt.colorbar()
    plt.scatter(train_X[:, 0], train_X[:, 1])
    plt.savefig("MLP_sample.png")
    plt.clf()

    plt.title("MLP predict")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.pcolor(xx, yy, y_predict.reshape([101, 101]), vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("MLP_predict.png")
