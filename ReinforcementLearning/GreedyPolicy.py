# coding:utf-8
# TD誤差によって状態価値関数を更新します
# 方策はグリーディー
import numpy as np
import matplotlib.pyplot as plt
import random

# フィールド
map = np.loadtxt("maze/maze_map.txt")
N, M = map.shape
# 過去の位置
x_prev, y_prev = N - 1, 0
# 状態価値関数
V = np.random.random([N, M]) - 0.5
# 報酬関数の定義
R = np.zeros_like(map) - 1
R[N - 1, M - 1] = 10
V[np.where(map == 0)] = 10
actionList = [[1, 0], [0, 1], [-1, 0], [0, -1]]

def draw():
    plt.clf()
    plt.imshow(V.T, interpolation = "nearest", origin = "lower")
    
    plt.plot(x, y, "ro")
    plt.colorbar()
    plt.xlim([0 - 0.5, N - 0.5])
    plt.ylim([0 - 0.5, M - 0.5])
    plt.draw()
    plt.pause(0.01)

def isWall(x, y):
    # 盤外判定
    if not (0 <= x <= N - 1 and 0 <= y <= M - 1):
        return True
    # 障害物判定
    if map[x,y]==0:
        return True
    return False
T = 0.1

def nextPos(x,y):
    ValueList = []
    # 次に移動することができる座標
    pos = [[x + dx, y + dy] for dx, dy in actionList if not isWall(x + dx, y + dy)]
    # 各座標の状態価値
    ValueList = [V[x, y] for x, y in pos]
    
    if len(ValueList) == 0:
        return ValueList[0]
    
    else:
        """
        # soft max
        T = np.exp(-cnt / 1000)
        if T < 0.01:
            T = 0.01
        ValueList = np.array(ValueList) / T
        ValueList = (ValueList - np.mean(ValueList)) 
        f = np.exp(ValueList) / np.sum(np.exp(ValueList))
        return pos[np.random.choice(range(len(f)), p = f)]
        """
        # epsiron greedy
        # if np.random.random() > 1 / (1 + np.exp(-cnt / 1000)):
        if np.random.random() > 0.99:
            return  random.choice(pos)
        else:
            return pos[np.argmax(ValueList)]

if __name__ == "__main__":
    # 学習パラメータ
    gamma, alpha = 0.99, 0.5
    # 収益,
    Revenue,r, cnt = 0, 0, 0
    # plt.ion()
    tmp = []
    while True:
        tmp.append(Revenue)
        x, y = nextPos(x_prev, y_prev)
        alpha = ((1 + np.exp(-cnt / 1000.0))) / 2
        if  cnt > 10 ** 5 and cnt % 1 == 0:

            draw()
        # 報酬を得る
        r = R[x, y]
        Revenue += r
        # 状態価値の更新
        V[x_prev, y_prev] += alpha * (r + gamma * V[x, y] - V[x_prev, y_prev])
        if r > 0 :
            x, y = 0, 0
        x_prev, y_prev = x, y
        cnt += 1