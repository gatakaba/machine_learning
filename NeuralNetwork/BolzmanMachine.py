# coding:utf-8
"""
Boltzman MachineはHopfield Networkに確率的挙動を追加したもの

実際に動かしてサンプリングすることによって遷移確率を推定できそう
"""
import numpy as np
import matplotlib.pyplot as plt

class BoltzmanMachine(object):
    def __init__(self, W, th):
        self.N = len(W)
        self.W, self.th = W, th
        #温度パラメータ
        self.c = 20
        self.x = np.random.choice([0, 1], size = self.N)
    def sigmoid(self,s):
        return 1/(1+np.exp(-s/self.T))
    def Run(self):
        cnt = 0
        while True:
            self.T = self.c / np.log(cnt + 1)
            print self.T
            # 変更する素子の決定
            i = np.random.choice(range(self.N - 1))
            s = np.dot(self.W[i], self.x) - self.th[i]
            p1 = self.sigmoid(s)
            
            if p1 > np.random.random():
                self.x[i] = 1
            else:
                self.x[i] = 0
            E = -np.dot(np.dot(self.x, self.W), self.x) / 2 + np.dot(self.th, self.x)
            print self.x, E
            cnt += 1
    

if __name__ == "__main__":
    
    W = np.array([[0, 4, -4],
                  [4, 0, -2],
                  [-4, -2, 0]])
    th = np.array([5, -4, -7])
    b = BoltzmanMachine(W, th)
    b.Run()