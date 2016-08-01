# coding:utf-8
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


digits = load_digits(10)
digits.data
digits.target


N = int(len(digits.data) * 0.7)

digits_train = digits.data[:N]
target_train = digits.target[:N]

digits_test = digits.data[N:]
target_test = digits.target[N:]

param_grid = [
  {'C': [10 ** i for i in range(-4, 3)], 'kernel': ['linear']},
  {'C': [10 ** i for i in range(-4, 3)], 'gamma': [10 ** i for i in range(-4, 3)], 'kernel': ['rbf']},
 ]

gscv = GridSearchCV(svm.SVC(verbose=True, probability=True), param_grid, cv=3, scoring="mean_squared_error")
gscv.fit(digits_train, target_train)
print gscv.best_estimator_ 
print gscv.score(digits_test, target_test)
print gscv.predict_proba(digits_test)

