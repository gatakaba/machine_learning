import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
np.random.seed(0)
N = 2000
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
xx, yy = np.meshgrid(x, y)
X = np.c_[np.ravel(xx), np.ravel(yy)]
np.random.shuffle(X)
# X = X[:2000]

tmp = []
cnt = 0
while len(tmp) < N:
    tmp.append(X[cnt])
    cnt += 1
X = np.array(tmp)

target = []
for i in range(N):
    if (X[i, 0] - 0.5) ** 2 + (X[i, 1] - 0.5) ** 2 <= 0.04:
        target.append(1)
    else:
        target.append((1 + X[i, 0]) / 2 * np.sin(6 * np.pi * X[i, 0] ** 0.5 * X[i, 1] ** 2) ** 2)
target = np.array(target)

plt.figure(0)
plt.scatter(X[:, 0], X[:, 1], c=target)
plt.colorbar()

# clf = SVR(kernel="rbf", gamma=100)
# clf = KNeighborsRegressor(n_neighbors = 5, weights = 'distance')
# clf = RandomForestRegressor()
clf = GradientBoostingRegressor(n_estimators=1000)
clf.fit(X, target)
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
xx, yy = np.meshgrid(x, y)
X = np.c_[np.ravel(xx), np.ravel(yy)]
z = clf.predict(X).reshape([1000, 1000])
plt.figure(1)
plt.imshow(z, interpolation="nearest", origin="lower", extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()

if type(clf) is tree.DecisionTreeRegressor:
    with open("DecisionTree.dot", 'w') as f:
         f = tree.export_graphviz(clf, out_file=f)
    f.close()
    import os
    os.system("dot -Tpng DecisionTree.dot -o DecisionTree.png")
