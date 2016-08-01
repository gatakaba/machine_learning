# coding:utf-8
"""
HMMによって二次元ガウス関数を生成
"""
import numpy as np
import matplotlib.pyplot as plt
N = 4
"""
# 遷移行列の定義
A = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]])
# パラメータ
muList = np.array([[-2, -1], [1, 1], [2, -2]])
sigmaList = np.array([[[0.1, 0], [0, 0.1]], [[0.1, -0.05], [-0.05, 0.2]], [[0.2, 0], [0, 0.1]]])
"""
# 遷移行列の定義
A = np.array([[0.1, 0.1, 0.1, 0.25],
              [0.1, 0.1, 0.1, 0.25],
              [0.1, 0.1, 0.1, 0.25],
              [0.7, 0.7, 0.7, 0.25] ])

# パラメータ
muList = np.array([[-2, -2], [0, 3.5], [2, -2], [0, 0]]) * 2
sigmaList = np.array([[[0.1, 0], [0, 0.1]], [[0.1, -0.05], [-0.05, 0.2]], [[0.2, 0], [0, 0.1]], [[0.05, 0], [0, 0.05]]])


# 隠れ変数
z = np.zeros(N)
z[0] = 1

tmp = []
for i in range(1000):
    index = np.random.choice(range(N), size = 1, p = np.dot(A, z))[0]
    z *= 0
    z[index] = 1
    X = np.random.multivariate_normal(muList[index], sigmaList[index])
    tmp.append(np.r_[X, index])
    
tmp = np.array(tmp)
cmap = plt.cm.jet
plt.scatter(tmp[:, 0], tmp[:, 1], color = cmap(tmp[:, 2] / 3.0))
plt.plot(tmp[:, 0], tmp[:, 1], alpha = 0.25)
plt.show()