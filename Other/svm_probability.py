# coding:utf-8

from sklearn import svm
from sklearn.datasets import load_digits
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

clf = svm.SVC(probability=True)

clf.fit(digits_train, target_train)
result = clf.predict_proba(digits_test)

plt.hist(np.max(result, axis=1))
# plt.pcolor(result)
# plt.colorbar()
plt.show()
