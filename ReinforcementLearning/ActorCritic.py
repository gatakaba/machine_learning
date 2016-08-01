# coding:utf-8
# ActorCriticの習作
import numpy as np
import matplotlib.pyplot as plt
import random

# フィールド
N, M = 100, 100
# 過去の位置
x_prev, y_prev = N / 2, M / 2
# 状態価値関数
V = np.random.random([N, M])
# 報酬関数の定義
R = np.zeros([N, M]) - 10
 
R[N / 2, M / 2] = 1

# 収益\
Revenue = 0.0
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
    if x < 0:
        return True
    elif x > N - 1 :
        return True
    if y < 0:
        return True 
    elif y > M - 1 :
        return True
    return False

actionList = [[1, 0], [0, 1], [2, 1], [1, 2], [2, -1], [1, -2], [10, 0]]

def nextPos(x,y):
    ValueList = []
    tmpList = []
    for action in  actionList:
        dx, dy = action
        if not isWall(x + dx, y + dy):
            tmpList.append([dx, dy])
        if not isWall(x - dx, y - dy):
            tmpList.append([-dx, -dy])
    for action in tmpList:
        dx, dy = action
        ValueList.append(V[x + dx, y + dy])
    
    # ランダム行動する確率を徐々に減らしてゆく
    if np.random.random() > 1 / (1 + np.exp(-cnt / 1000)):
        dx, dy = random.choice(tmpList)
    else:
        dx, dy = tmpList[np.argmax(ValueList)]
    return x + dx, y + dy


if __name__ == "__main__":
    xx, yy = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, N))
    plt.ion()
    # 学習率
    alpha = 0.3
    cnt = 0
    gamma = 0.99
    r = 0
    while True:
        x, y = nextPos(x_prev, y_prev)
        alpha = (1 + np.exp(-cnt / 1000.0))
        if  cnt > 10 ** 5 and cnt % 1 == 0:

            draw()
        # 報酬を得る
        r = R[x, y]
        Revenue += r
        # 状態価値の更新
        delta = r + gamma * V[x, y] - V[x_prev, y_prev]
        
        
        if r > 0 or r < -20:
            x, y = N / 2, M / 2
            x = np.random.randint(N - 1)
            y = np.random.randint(M - 1)
            # R = np.zeros_like(R) - 10
            # R[N / 2, M / 2 + np.random.randint(0, 2)] = 10
        x_prev, y_prev = x, y
        cnt += 1