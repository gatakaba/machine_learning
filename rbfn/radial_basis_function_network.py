# coding:utf-8
"""
RBFNの実装

改良の余地あり

"""
import numpy as np
from utilities import test_function_nonaka
import matplotlib.pyplot as plt


class RBFN(object):
    def __init__(self, is_fit_intercept=False):
        self.n_samples = None
        self.n_features = None
        self.w = None
        self.beta = 1

    def phi(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) * self.beta)

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.train_data = X
        design_matrix = np.empty([self.n_samples, self.n_samples])

        for i in range(self.n_samples):
            for j in range(self.n_samples):
                design_matrix[i, j] = self.phi(X[i, :], X[j, :])

        self.w = np.dot(np.linalg.pinv(design_matrix), y)
        return self

    def predict(self, X):
        target_List = []
        for x in X:
            phi_list = []
            for i in range(self.n_samples):
                phi_list.append(self.phi(x, self.train_data[i]))
            Phi = np.array(phi_list)
            target_List.append(np.dot(Phi, self.w))
        y = np.array(target_List)
        return y


if __name__ == "__main__":
    train_X = np.random.uniform(0, 1, size=[100, 2])
    train_y = test_function_nonaka(train_X)
    train_y = train_X[:, 0] + train_X[:, 1]
    clf = RBFN()
    clf.fit(train_X, train_y)
    # print(clf.score(train_X, train_y))


    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X = np.c_[np.ravel(xx), np.ravel(yy)]
    prediction_y = clf.predict(X)
    plt.pcolor(xx, yy, prediction_y.reshape([100, 100]))
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
    plt.show()
