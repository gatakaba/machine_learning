# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

class NeuroDynamics(object):
    def __init__(self):
        np.random.seed(0)
        # 素子数・パターン数
        self.N = 100
        # 内部状態を乱数で決定
        self.u = np.random.normal(size = self.N)
        # 結合係数行列
        self.W = np.random.normal(size = [self.N, self.N])
        # 作動係数
        self.tau = 10
        # 学習係数1
        self.tau_dash = 5000 * self.tau
        # 学習係数2
        self.alpha = 10
        # 時間刻み
        self.dt = 1
        # 出力値計算
        self.y = self.func(self.u)
        # 重みベクトル初期化
        self.dW = np.zeros([self.N, self.N])
        self.makeTeach()
        
    def func(self,u):
        # 非単調増加関数
        k , c, c_dash, h = -1, 50, 10, 0.5
        u[np.where(u > 5)] = 5
        u[np.where(u < -5)] = -5 
        tmp1 = (1 - np.exp(-c * u)) / (1 + np.exp(-c * u))
        tmp2 = (1 + k * np.exp(c_dash * (np.abs(u) - h))) / (1 + np.exp(c_dash * (np.abs(u) - h)))
        return tmp1 * tmp2
    def makeTeach(self):
        # 学習用パターン列作成
        x = np.ones(self.N)
        x[:self.N / 2.0] *= -1
        tmp = []
        for i in range(self.N / 2):
            tmp.append(np.roll(x, i))
        self.R = np.array(tmp)
        
    def learn(self):
        # 学習
        for cnt in range(100 * 2):
            for r in self.R:
                dW = (-self.W + self.alpha * np.dot(r[np.newaxis].T, self.y[np.newaxis])) / self.tau_dash * self.dt
                self.W += dW
                self.y = self.func(self.u)
                du = (-self.u + np.dot(self.W, self.y) + 100 * r) / self. tau * self.dt
                self.u += du
            
    def draw(self):
        # 学習した結果を描画
        tmp = []
        u = np.copy(self.R[0])
        tmp.append(np.sign(np.copy(u)))
        for i in range(10 ** 3):
        # for r in self.R:
            y = self.func(u)
            # du = (-u + np.dot(self.W, y) + 100 * r) / self.tau * self.dt
            du = (-u + np.dot(self.W, y)) / self.tau * self.dt
            u += du
            tmp.append(np.copy(np.sign(u)))
        plt.figure(0)
        plt.pcolor(np.array(tmp))
        # plt.figure(1)
        # plt.pcolor(self.R)
        plt.show()
if __name__=="__main__":
    n = NeuroDynamics()
    n.learn()
    print "finish learn"
    n.draw()
    