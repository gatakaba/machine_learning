# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# データ作成
N = 1000
x1 = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], size = N)
x2 = np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], size = N)
x3 = np.random.multivariate_normal([-1, 1], [[0.1, 0], [0, 0.1]], size = N)
x4 = np.random.multivariate_normal([1, -1], [[0.1, 0], [0, 0.1]], size = N)

X = np.r_[x1, x2, x3, x4]

y = np.zeros(len(X))
y[2 * N:] = 1
y[:2 * N] = -1
# 識別開始
clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_leaf = 100).fit(X, y)

cmap = plt.cm.get_cmap("rainbow")

# ファイルを作成
with open("xor_condition.dot", 'w') as f:
     f = tree.export_graphviz(clf, out_file = f)
f.close()
import os
os.system("dot -Tpng xor_condition.dot -o xor_condition.png")

# 境界面を描画
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# plt.contour(xx, yy, Z.reshape([100, 100]))
plt.contourf(xx, yy, Z.reshape([100, 100]))
plt.scatter(X[:, 0], X[:, 1], c = cmap(y))
plt.savefig("xor_result.jpg", dpi = 300)
plt.show()

