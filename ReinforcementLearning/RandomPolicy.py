# coding:utf-8
# TD誤差によって状態価値関数を更新します
# 方策はランダム
import numpy as np
import matplotlib.pyplot as plt
import random

def drawPos(x, y):
    X = np.zeros([N, M])
    X[x, y] = 1
    plt.imshow(X, interpolation = "nearest")
    # plt.draw()
def draw():
    """
    エージェントの位置をfigure1,状態価値をfigure2に描画します
    """
    
    pass
def nextPos(x,y):
    dx, dy = random.choice([[1, 0], [-1, 0], [0, 1], [0, -1]])
    x += dx
    y += dy
    if x <= 0:
        x = 0
    elif x >= N - 1 :
        x=N-1
    if y <= 0:
        y = 0
    elif y >= M - 1 :
        y = M - 1
    return x, y

if __name__ == "__main__":
    # フィールド
    N, M = 20, 20
    # 現在位置
    x_prev, y_prev = N / 2, M / 2
    # 状態価値関数
    V = np.random.random([N, M])
    # 報酬関数の定義
    R = np.zeros([N, M])
    xx, yy = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, N))
    
    R[N / 2, M / 2] = 10
    
    plt.ion()
    # 学習率
    alpha = 0.05
    cnt = 0
    gamma = 0.9
    
    while True:
        if cnt % 100 == 0:
            plt.figure(0)
            drawPos(x_prev, y_prev)
            plt.pause(0.01)
            plt.figure(1)
            plt.clf()
            plt.imshow(V, interpolation = "nearest")
            
            plt.colorbar()
            plt.draw()
            plt.pause(0.01)
            
        x, y = nextPos(x_prev, y_prev)
        
        # 報酬を得る
        r = R[x, y]
        # 状態価値の更新
        V[x_prev, y_prev] = V[x_prev, y_prev] + alpha * (r + gamma * V[x, y] - V[x_prev, y_prev])
        if r > 5:
            x, y = 0, 0
        x_prev, y_prev = x, y
        cnt += 1