# coding:utf-8
"""
SDNN

"""
import numpy as np

from base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from adaptive_pattern_coding import SelectiveDesensitization


class AdaptiveSDNN(BaseEstimator):
    def __init__(self, hidden_layer_num=10, eta=10 ** -3, verbose=False):
        self.W = None
        self.n_samples, self.n_features = None, None
        self.eta = eta
        self.hidden_layer_num = hidden_layer_num
        self.verbose = verbose
        self.a = 3 / self.hidden_layer_num
        self.b = -1.5
        self.pattern_manager = SelectiveDesensitization(binary_vector_dim=200, division_num=3, reversal_num=1)

    def hidden_function(self, x):
        return (np.sign(x) + 1) / 2.0

    def activate_function(self, x):
        y = self.a * x + self.b
        return y

    def inverse_activate_function(self, y):
        x = (y - self.b) / self.a
        return x

    def _search_index(self, a, n_target, n_predict):
        """
        n_target > n_predictの場合、(n_target-n_predict)個のパーセプトロンを1が出るように修正
        修正するパーセプトロンは0以下の内、内部電位が高いパーセプトロン

        n_target < n_predictの場合、(n_predict-n_target)個のパーセプトロンを0が出るように修正
        修正するパーセプトロンは0以上の内、内部電位が低いパーセプトロン

        """
        error_num = int(round(np.abs(n_target - n_predict)))
        if error_num == 0:
            return []
        elif n_target > n_predict:
            negative_perceptron_values = np.sort(a[a < 0])[::-1]
            if len(negative_perceptron_values) > error_num:
                fix_perceptron_values = negative_perceptron_values[:error_num]
            else:
                fix_perceptron_values = negative_perceptron_values
        else:
            positive_perceptron_values = np.sort(a[a > 0])
            if len(positive_perceptron_values) > error_num:
                fix_perceptron_values = positive_perceptron_values[:error_num]
            else:
                fix_perceptron_values = positive_perceptron_values

        index_list = []
        for fix_perceptron_value in fix_perceptron_values:
            index = np.where(a == fix_perceptron_value)[0][0]
            index_list.append(index)
        return index_list

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        self.X_train_, self.y_train_ = np.copy(X), np.copy(y)

        upper = 1.2 * np.max(self.y_train_) - 0.2 * np.min(self.y_train_)
        lower = -0.2 * np.max(self.y_train_) + 1.2 * np.min(self.y_train_)
        self.a = upper - lower
        self.b = lower

        self.pattern_manager.make_binary_vector_tables(X)
        patterned_X = self.pattern_manager.convert_to_sd_pattern(X)
        n_samples, n_features = patterned_X.shape

        self.W = np.random.normal(0, 1, size=[self.hidden_layer_num, n_features])
        for j in range(50):
            for i in (range(n_samples)):
                # feedforward
                a = np.dot(self.W, patterned_X[i])
                z = self.hidden_function(a)
                n_predict = np.sum(z)
                n_target = self.inverse_activate_function(y[i])

                # 修正するパーセプトロンを選択
                index_list = self._search_index(a, n_target, n_predict)
                if not len(index_list) == 0:
                    self.W[index_list, :] += self.eta * np.sign(n_target - n_predict) * patterned_X[i]
                    # self.a -= (self.a * n_predict + self.b - y[i]) * n_predict * 10 ** -6
                    # self.b -= (self.a * n_predict + self.b - y[i]) * 10 ** -6

            if self.verbose:
                print(j, self.score(self.X_train_, self.y_train_))
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_train_", "y_train_"])
        prediction_list = []
        patterned_X = self.pattern_manager.convert_to_sd_pattern(X)
        for patterned_x in patterned_X:
            a = np.dot(self.W, patterned_x)
            z = self.hidden_function(a)
            a2 = np.sum(z)
            prediction = self.activate_function(a2)
            prediction_list.append(prediction)
        y = np.ravel(prediction_list)
        return y
