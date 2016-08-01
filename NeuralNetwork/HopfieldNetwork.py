# coding:utf-8

# 非同期ホップフィールドネットワーク
# 入力したベクトルを学習して記憶する機能と
# 提示したベクトルよりデータを復元する想起をする機能を備える

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

class HopfieldNeuralNetwork(object):
    """
    ホップフィールドニューラルネットワーククラス
    入力用データ行列がクラス作成時に必要
    
    """
    def __init__(self):
        pass
    def fit(self, X):
        """
        学習メソッド
        重み行列の作成
        
        X:正則化されたデータ行列
        """
        self.dataDim, self.dataNum = X.shape
        # ネットワーク結合行列の作成
        self.W = np. dot(X, X.T)
        # 自己結合係数を0に
        self.W[np.diag_indices(self.dataDim)] = 0
    def transform(self, x):
        """
        想起開始
        
        x:入力ベクトル
        """
        for i in range(10):
            print -0.5 * np.dot(np.dot(x, self.W), x)
            
            x = np.sign(np.dot(self.W, x))
            np.dot(np.dot(x, self.W), x)
        return x

def MNIST():
    digits = load_digits(2)
    X = digits.data.T
    
    X=np.sign(X-8)
    x = np.sign(np.random.normal(size=X.shape[0]))
    h = HopfieldNeuralNetwork()
    h.fit(X)
    x = h.transform(x)
    
    plt.title("network data")
    plt.pcolor(x.reshape(8, 8))
    plt.show()
    
def image():
    from scipy import misc
    # データの読み込み
    tuki = np.mean(misc.imread("image/tuki.jpg"), axis=2)
    yasuna = np.mean(misc.imread("image/yasuna.jpg"), axis=2)
    ika = np.mean(misc.imread("image/ika.jpg"), axis=2)
    x1 = np.ravel(tuki)
    x2 = np.ravel(yasuna)
    x3 = np.ravel(ika)
    X = np.array([x1, x2, x3]).T
    # データの正規化
    X = np.sign((X - 128.0) / 256.0)
    # ネットワーク結合強度を計算
    W = np.dot(X, X.T)
    # 自己結合係数を0に
    W[np.diag_indices(W.shape[0])] = 0
    print W.shape
    print np.size(W)
    # ランダムな画像を提示して想起させる
    for j in range(10):
        x = np.random.normal(size=len(x1))
        for i in range(3):
            x = np.sign(np.dot(W, x))
        plt.imshow(x.reshape(tuki.shape))
        plt.colorbar()
        plt.savefig(str(j) + ".jpg")
        plt.clf()
if __name__ == "__main__":
    MNIST()
    x1 = [1, 1, -1, -1]
    x2 = [-1, -1, 1, 1]

    X = np.array([x1, x2]).T
    h = HopfieldNeuralNetwork()
    h.fit(X)
    x = [-1, -1, -1, 1]
    print h.transform(x)
