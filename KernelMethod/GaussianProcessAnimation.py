# coding:utf-8
# ガウス過程
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    # return p[0] * np.sin(2 * np.pi * x * p[1] + p[2]) + p[3] * np.sin(2 * np.pi * x * p[4] * 3 + p[5]) + p[6] * x
    return x / 2 + np.sin(2 * np.pi * x) + np.sin(2 * np.pi * x * 3)
     
def kernel(x1,x2,th):
    return th[0] * np.exp(-th[1] / 2 * np.dot((x1 - x2), (x1 - x2))) + th[2] + th[3] * np.dot(x1, x2)

def relateVector(sample_x, x):
    # 既存データと新しい値の関連ベクトルを作成
    k = np.zeros(len(sample_x))
    for i, x1 in enumerate(sample_x):
        k[i] = kernel(x1, x, th)
    return k

if __name__ == "__main__":
    N = 10
    sample_x = np.linspace(0, 1, N)
    sample_x = np.random.random(N) 
    sample_x.sort()
    
    plt.ion()
    while True:
        myList = [0]
        p = np.random.random(size = 10)
        L = 50
        e = np.random.normal(loc = 0, scale = 0.5, size = L * 2)
        for i in range(50):
            myList.append(np.random.random() * 1.9)
            sample_x = np.array(myList)
            beta = 10
            N = len(sample_x)
            sample_t = func(sample_x) + e[:i + 2]
            
            th = [1, 64, 1, 1]
            # カーネル行列の作成
            K = np.zeros([N, N])
            for i, x1 in enumerate(sample_x):
                for j, x2 in enumerate(sample_x):
                    K[i, j] = kernel(x1, x2, th)
            K += np.eye(N) / beta
            
            K_inv = np.linalg.inv(K)
            mList, stdList = [], []
            plot_x = np.linspace(0, 2, 200)
            
            # 各点における平均値,分散を計算
            for x in plot_x:
                k = relateVector(sample_x, x)
                c = kernel(x, x, th) + 1 / beta
                mList.append(np.dot(np.dot(k, K_inv), sample_t))
                stdList.append(c - np.dot(np.dot(k, K_inv), k))
            mList = np.array(mList)
            stdList = np.array(stdList) ** 0.5
            
            # 予想分布を描画
            # plt.plot(plot_x, mList, "k", linewidth = 3)
            plt.fill_between(plot_x, mList - stdList, mList + stdList, facecolor = 'gray', alpha = 0.25)
            # サンプルデータを描画
            # plt.plot(plot_x, func(plot_x), "g--")
            plt.plot(sample_x, sample_t, "go")
            plt.xlim([np.min(plot_x), np.max(plot_x)])
            plt.ylim([np.min(func(plot_x)) , np.max(func(plot_x)) ])
            
            # plt.savefig("%02d" % i + ".png")
            plt.draw()
            plt.pause(0.1)
            plt.clf()