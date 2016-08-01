# coding:utf-8
"""
2015/1/27
多次元入力多次元出力の関数を学習する
"""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

import matplotlib.pyplot as plt
import mySignal
import numpy as np


if __name__=="__main__":
    data = np.loadtxt("index.CSV", delimiter=",")
    data = data[:11000, :]
    data = mySignal.ToIEMG(data)
    data = mySignal.SMA(data)
    
    N = len(data) - 1000
    M = 10
    train_X = data[:N, :M]
    train_t = data[:N, M:]
    
    test_X = data[N:, :M]
    test_t = data[N:, M:]
    
    # clf = RandomForestRegressor(n_estimators=10)
    # clf = KNeighborsRegressor(n_neighbors=10)
    # clf = LinearRegression()
    # clf = DecisionTreeRegressor()
    # clf = GaussianProcess(theta0=10)
    clf = ExtraTreeRegressor()
    clf.fit(train_X, train_t)
    
    pre_t = clf.predict(test_X)
    plt.plot(pre_t)
    plt.plot(test_t)

    plt.show()
