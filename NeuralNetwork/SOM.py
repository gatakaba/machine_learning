# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def axisList(N):
    myList = []
    for i in range(-N, N + 1):
        for j in range(int(-(N ** 2 - i ** 2) ** 0.5), int((N ** 2 - i ** 2) ** 0.5) + 1):
            myList.append([i, j])
    return myList
def distance(px, py):
    return 0.5 / (1.0 + px ** 2 + py ** 2) * 0.01

N = 30
np.random.seed(0)
W = np.random.uniform(low=0, high=255, size=[N, N, 3])
# W *= 0
# W[5, 5, :] = [255]

axisList = axisList(2)
# plt.ion()
for cnt in range(10 ** 4):
    # x = np.array([128, 128, 128])
    x = np.random.uniform(low=0, high=255, size=3)
    
    distanceMat = np.sum(np.abs(W - x), axis=2)
    i, j = np.where(distanceMat == distanceMat.min())
    
    for p in axisList:
        px = (i + p[0]) % N
        py = (j + p[1]) % N
        W[ px, py, :] = W[ px, py, :] + distance(p[0], p[1]) * (x - W[ px, py, :])
    
    plt.ion()
    plt.imshow(W, interpolation='nearest')
    plt.draw()
    plt.pause(0.01)
    plt.clf()
    
plt.imshow(W, interpolation='nearest')
plt.show()

  