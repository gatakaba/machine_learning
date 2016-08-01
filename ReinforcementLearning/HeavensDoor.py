# coding:utf-8
# TD誤差によって状態価値関数を更新します
# 方策はグリーディー
import numpy as np
import matplotlib.pyplot as plt
import random

# フィールド
map = np.array([[1, -1, -1, -1, 1],
                [0, 0, -1, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, -1, 0, 0],
                [-1, -1, -1, 0, 0]])
N, M = map.shape
# 過去の位置
x_prev, y_prev = N - 1, 0
# 状態価値関数
V = np.random.random([N, M])
# 報酬関数の定義
R = map
V[np.where(map == 0)] = 10
# 収益
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
    # 盤外判定
    if x < 0:
        return True
    elif x > N - 1 :
        return True
    if y < 0:
        return True 
    elif y > M - 1 :
        return True
    # 障害物判定
    if map[x, y] == 0:
        return True
    return False

# actionList = [[1, 0], [0, 1], [2, 1], [1, 2], [2, -1], [1, -2]]
actionList = [[1, 0], [0, 1]]
def nextPos(x, y):
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
    # if np.random.random() > 1 / (1 + np.exp(-cnt / 1000)):
    if np.random.random() > 0.99:
        dx, dy = random.choice(tmpList)
    else:
        dx, dy = tmpList[np.argmax(ValueList)]
    return x + dx, y + dy

if __name__ == "__main__":
    xx, yy = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, N))
    plt.ion()
    # 学習率
    alpha = 0.5
    cnt = 0
    gamma = 0.99
    r = 0
    while True:
        x, y = nextPos(x_prev, y_prev)
        alpha = ((1 + np.exp(-cnt / 1000.0))) / 2
        if  cnt > 10 ** 5 and cnt % 1 == 0:
            #print cnt, Revenue
            draw()
        # 報酬を得る
        r = R[x, y]
        Revenue += r
        
        # 状態価値の更新
        V[x_prev, y_prev] += alpha * (r + gamma * V[x, y] - V[x_prev, y_prev])
        if r > 0 :
            # x, y = N - 1, 0
            x, y = 0, 0
        x_prev, y_prev = x, y
        cnt += 1