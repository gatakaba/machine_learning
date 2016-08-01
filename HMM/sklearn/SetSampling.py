# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import hmm

startprob = np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
covars = np.tile(np.identity(2), (3, 1, 1))

model = hmm.GaussianHMM(3, "full", startprob, transmat)
model.means_ = means
model.covars_ = covars
X, Z = model.sample(1000)
cmap = plt.cm.jet

plt.scatter(X[:, 0], X[:, 1], color = cmap(Z / 3.0))
plt.plot(X[:, 0], X[:, 1], alpha = 0.25)
plt.show()