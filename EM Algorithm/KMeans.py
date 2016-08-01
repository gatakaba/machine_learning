# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
# パラメータの決定
N1, N2, N3 = 1000, 1000, 100
N = N1 + N2 + N3
K = 3
# 色スケール
# colorScale = np.array([2 ** (i - K) for i in range(K)])
colorScale = np.random.random(size=K) * 2
# 色マップ
cmap = plt.cm.jet
# データ点作成
x1 = np.random.multivariate_normal([-0.5, -0.5], [[0.1, 0.05], [0.05, 0.1]], N1)
x2 = np.random.multivariate_normal([0.5, 0.5], [[0.1, 0.05], [0.05, 0.1]], N2)
x3 = np.random.multivariate_normal([0.5, -0.5], [[0.1, 0.05], [0.05, 0.1]], N3)
X = np.r_[x1, x2, x3]

r = np.zeros([N, K])
# 初期セントロイド作成
centroids = []
for k in range(K):
    centroids.append(np.random.random([2]))
plt.ion()
for i in range(100):
    # セントロイドと距離を計算
    # 行:サンプル 列:クラス
    d = []
    for k in range(K):
        d.append(np.diag(np.dot((X - centroids[k]), (X - centroids[k]).T)))
    D = np.array(d).T
    # n番目のデータの割り振り修正
    index = np.argmin(D, axis=1)
    # 割り振り行列に変換
    r *= 0
    for n, i in enumerate(index):
        r[n, i] = 1
    
    # データ点描画
    plt.scatter(X[:, 0], X[:, 1], c=cmap(np.dot(r, colorScale)))
    # セントロイド描画
    for k in range(K):
        plt.scatter(centroids[k][0], centroids[k][1], marker="*", s=200)
    plt.draw()
    plt.pause(0.1)
    plt.clf()

    # セントロイドの更新
    for k in range(K):
        centroids[k] = np.dot(r[:, k] , X) / np.sum(r[:, k])
