# coding:utf-8

# 混合ガウス分布のEMアルゴリズム

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

tmp=[]
for i in range(2, 10):
    g = mixture.GMM(n_components = i, n_iter = 10 ** 2, covariance_type = "full")
    g.fit(X)
    tmp.append(g.aic(X))
plt.plot(range(2, 10), np.array(tmp))
# plt.show()
plt.ion()
plt.draw()
plt.pause(0.01)
#print g.means_.shape, g.covars_.shape, g.weights_.shape

from sklearn.decomposition import PCA
pca = PCA(3)
X = pca.fit_transform(X)
from mayavi import mlab
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1) ]

for i, color in enumerate(colors):
    index = np.where(iris.target == i)
    mlab.points3d(X[index, 0], X[index, 1], X[index, 2], color = color, resolution = 30, scale_factor = 0.3)

mlab.show()

