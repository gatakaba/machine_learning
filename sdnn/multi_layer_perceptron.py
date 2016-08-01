# coding:utf-8
"""
mlp
TODO : コメントを加える、終了条件について検討する
"""
import numpy as np
from base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted


class MultiLayerPerceptronRegression(BaseEstimator):
    def __init__(self, hidden_layer_num=100, eta=0.01, verbose=False):
        self.W1 = None
        self.W2 = None
        self.eta = eta
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose

    def hidden_function(self, x):
        return np.tanh(x)

    def hidden_differential_function(self, x):
        return 1 - self.hidden_function(x) ** 2

    def activate_function(self, x):
        return x

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)
        n_samples, n_features = X.shape
        intercepted_X = BaseEstimator.add_columns(X)
        self.W1 = np.random.normal(0, 1, size=[self.hidden_layer_num, n_features + 1])
        self.W2 = np.random.normal(0, 1, size=[1, self.hidden_layer_num + 1])
        for j in range(1000):
            for i in range(n_samples):
                # feedforward
                a1 = np.dot(self.W1, intercepted_X[i])
                z = self.hidden_function(a1)
                intercepted_z = np.r_[1, z]
                a2 = np.dot(self.W2, intercepted_z)
                prediction = self.activate_function(a2)
                # backpropagete
                delta2 = (prediction - y[i])
                dW2 = np.outer(delta2, intercepted_z)
                delta1 = np.dot(self.W2.T, delta2)[1:] * self.hidden_differential_function(a1)
                dW1 = np.outer(delta1, intercepted_X[i])
                # update weight matrix
                self.W2 -= self.eta * dW2
                self.W1 -= self.eta * dW1
            if self.verbose:
                print(j, self.score(self.X_train_, self.y_train_))
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_train_", "y_train_"])
        prediction_list = []
        if X.ndim == 1:
            X = np.atleast_2d(X)
        intercepted_X = BaseEstimator.add_columns(X)
        for intercepted_x in intercepted_X:
            h = np.dot(self.W1, intercepted_x)
            z = self.hidden_function(h)
            intercepted_z = np.r_[1, z]
            y = self.activate_function(np.dot(self.W2, intercepted_z))
            prediction_list.append(y)
        y = np.ravel(prediction_list)
        return y
