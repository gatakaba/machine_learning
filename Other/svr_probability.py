# coding:utf-8

from sklearn import svm
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np


x = np.random.uniform(0, 10, size=5000)
t = np.sin(2 * np.pi * x)
y = t + np.random.normal(0, 1, size=5000)

N = int(len(x) * 0.7)

x_train = x[:N]
y_train = y[:N]

x_test = x[N:]
y_test = y[N:]

index = np.argsort(x_test)
x_test = x_test[index]
y_test = y_test[index]

param_grid = [
  {'C': [2 ** i for i in range(0, 3)], 'kernel': ['linear']},
  {'C': 2 ** np.linspace(4.5, 5.5, 10), 'gamma': 2 ** np.linspace(0.5, 1.5, 10), 'kernel': ['rbf']},
 ]

gscv = GridSearchCV(svm.SVR(), param_grid, cv=5, scoring="mean_squared_error", verbose=True, n_jobs=-1)
gscv.fit(x_train[None].T, y_train)

print gscv.best_estimator_
print gscv.best_params_
print gscv.best_score_
print gscv.score(x_train[None].T, y_train)
print gscv.score(x_test[None].T, y_test)

plt.plot(x_test, y_test)
plt.plot(x_test, np.sin(2 * np.pi * x_test))
plt.plot(x_test, gscv.predict(x_test[None].T))
plt.show()


