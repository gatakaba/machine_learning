# coding:utf-8
"""
MNIST+PCA
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def PCA2D():
    N = 5
    digits = load_digits(n_class = N)
    data, target = digits.data, digits.target
    pca = PCA(2, whiten = True)
    print pca.explained_variance_ratio_
    X = pca.fit_transform(data)
    cmap = plt.cm.jet
    c = cmap(target * 1.0 / N)
    
    for i in range(N):
        index = np.where(target == i)
        plt.scatter(X[index, 0], X[index, 1], c = c[index], label = str(i))
    plt.legend()
    plt.show()

def PCA3D():
    N = 4
    digits = load_digits(n_class = N)
    data, target = digits.data, digits.target
    pca = PCA(3, whiten = True)
    X = pca.fit_transform(data)
    print pca.explained_variance_ratio_
    cmap = plt.cm.jet   
    
    from mayavi import mlab
    for i in range(N):
        index = np.where(target == i)
        mlab.points3d(X[index, 0], X[index, 1], X[index, 2], color = cmap(i * 1.0 / N)[:3], scale_factor = 0.25)
        
    mlab.show()
PCA3D()