# coding:utf-8

# 混合ガウス分布のEMアルゴリズム

import numpy as np
import matplotlib.pyplot as plt

class data():
    pass

def gaussian(x, mean, cov):
    """多変量ガウス関数"""
    temp1 = 1 / ((2 * np.pi) ** (x.size / 2.0))
    temp2 = 1 / (np.linalg.det(cov) ** 0.5)
    temp3 = -0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean)
    return temp1 * temp2 * np.exp(temp3)

def likelihood(X, means, cov, pi):
    """対数尤度関数"""
    sum = 0.0
    for n in range(len(X)):
        temp = 0.0
        for k in range(K):
            temp += pi[k] * gaussian(X[n], mean[k], cov[k])
        sum += np.log(temp)
    return sum

def draw(X, mean, cov, pi):
    N = 100
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, N), np.linspace(-2.5, 2.5, N))
    Xplot = np.c_[np.ravel(xx), np.ravel(yy)]
    z1, z2 = np.empty(N ** 2), np.empty(N ** 2)
    
    for i in range(N ** 2):
        z1[i] = gaussian(Xplot[i], mean[0], cov[0])
        z2[i] = gaussian(Xplot[i], mean[1], cov[1])

    plt.clf()
    # 等高線描画
    plt.contour(xx, yy, z1.reshape((N, N)))
    plt.contour(xx, yy, z2.reshape((N, N)))
    # サンプル点描画
    plt.scatter(X[:, 0], X[:, 1], c = gamma[:, 0])
    # 平均値描画
    plt.scatter(*mean[0], c = "r", marker = "*", s = 200)
    plt.scatter(*mean[1], c = "r", marker = "*", s = 200)
    plt.draw()
    plt.pause(0.01)
    
if __name__ == "__main__":
    class data():
        pass
    d1 = data()
    d2 = data()
    d1.N, d1.mu, d1.sigma = 400, np.array([-0.5, 0.5]) , np.array([[0.5, 0.05], [0.05, 0.1]])
    d2.N, d2.mu, d2.sigma = 400, np.array([0.5, -0.5]) , np.array([[0.5, -0.05], [-0.05, 0.1]])
    # 訓練データ
    x1 = np.random.multivariate_normal(d1.mu, d1.sigma, d1.N)
    x2 = np.random.multivariate_normal(d2.mu, d2.sigma, d2.N)
    X = np.r_[x1, x2]
    # 訓練データ数
    N = d1.N + d2.N
    # クラス数
    K = 2
    # 平均、分散、混合系数を初期化
    mean = np.random.random((K, 2))
    cov = np.zeros((K, 2, 2)) 
    
    for k in range(K):
        cov[k] = [[1.0, 0.0], [0.0, 1.0]]
    pi = np.random.rand(K)
    # 負担率の空配列を用意
    gamma = np.zeros((N, K))
    # 対数尤度の初期値を計算
    like = likelihood(X, mean, cov, pi)

    turn = 0
    plt.ion()
    while True:
        print turn, like
        # E-step : 現在のパラメータを使って、負担率を計算
        for n in range(N):
            # 分母はkによらないので最初に1回だけ計算
            denominator = 0.0
            for j in range(K):
                denominator += pi[j] * gaussian(X[n], mean[j], cov[j])
            # 各kについて負担率を計算
            for k in range(K):
                gamma[n][k] = pi[k] * gaussian(X[n], mean[k], cov[k]) / denominator
        
        # M-step:現在の負担率を使ってパラメータを再計算
        for k in range(K):
            # Nkを計算
            Nk = np.sum(gamma, axis = 0)[k]
            # 平均を計算
            mean[k] = np.zeros(2)
            for n in range(N):
                mean[k] += gamma[n][k] * X[n]
            
            mean[k] /= Nk
            
            # 共分散を計算
            cov[k] = np.zeros([2, 2])
            for n in range(N):
                temp = X[n] - mean[k]
                cov[k] += gamma[n][k] * np.matrix(temp).reshape(2, 1) * np. matrix(temp).reshape(1, 2)
            cov[k] /= Nk
            # 混合係数を再計算
            pi[k] = Nk / N
        # 収束判定
        new_like = likelihood(X, mean, cov, pi)
        diff = new_like - like
        if diff < 0.01 :
            # break
            pass
        like = new_like
        turn += 1
        draw(X, mean, cov, pi)