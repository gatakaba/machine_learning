# coding:utf-8
"""
Solve TSP using SOM by アンジェニオール
"""
import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, x):
        self.location = x
        self.captureNum = 0 
        self.lifeTime = 0
def plot(cityList, NodeList):
    for i in range(len(NodeList)):
        p1 = [NodeList[i - 1].location[0], NodeList[i].location[0]]
        p2 = [NodeList[i - 1].location[1], NodeList[i].location[1]]
        plt.plot(p1, p2, "bo-")
        plt.scatter(cityList[:, 0], cityList[:, 1])
    
    plt.pause(0.01)
    plt.clf()
    plt.draw()
    # plt.show()
    
if __name__ == "__main__":
    def winner(city, NodeList):
        # 勝者の決定・ノードを近づける
        min = -1
        index = -1
        for i, node in enumerate(NodeList):
            l = np.sum(np.abs(node.location - city))
            if min < 0 or min > l:
                min = l
                index = i
        return index, l
        
    # 都市・ノードの初期化
    # np.random.seed(1)
    N = 100
    cityList = np.random.random([N, 2])
    
    NodeList = []
    NodeList.append(Node(np.random.random(2)))
    NodeList.append(Node(np.random.random(2)))
    NodeList.append(Node(np.random.random(2)))
    plt.ion()
    
    # 利得
    a1 = 0.01
    a2 = 0.25 * a1 
    cnt = -1
    while True:
        cnt += 1
        print cnt
        if cnt % 10 == 0:
            plot(cityList, NodeList)            
        for city in cityList:
            # 勝者ノード決定
            index, l = winner(city, NodeList)
            # lifeTime,captureNum更新
            NodeList[index].lifeTime = 0
            NodeList[index].captureNum += 1
            # 勝者を都市に近づける
            NodeList[index].location += a1 * (city - NodeList[index].location) * l
            # 勝者近傍も近づける
            index1 = (index + 1) % len(NodeList)
            index2 = (index - 1)
            NodeList[index1].location += a2 * (city - NodeList[index1].location) * l
            NodeList[index2].location += a2 * (city - NodeList[index2].location) * l
        # すべての都市がノードを持つ場合終了
            # captureNumがすべて1であるか
        import copy
        tmpList = copy.deepcopy(NodeList)
        # ノードの複製・削除
        # captureNumが2以上ならばリストに追加
        j = 0
        for i, node in enumerate(tmpList):
            if node.captureNum >= 2 :
                # print "create Node:" + str(i)
                NodeList.insert(i + j, Node((NodeList[i + j - 1].location + NodeList[i + j].location) / 2))
                j += 1
        tmpList = copy.deepcopy(NodeList)
        # 生存時間が規定値以上ならば削除
        j = 0
        for i, node in enumerate(tmpList):
            if node.lifeTime >= 3:
                # print "delete Node:" + str(i)
                NodeList.pop(i - j)
                j += 1
        # 利得パラメータの更新
        # a1, a2 = a1 * (1 - 0.0001), a2 * (1 - 0.0001)
        # captureNum初期化
        for node in NodeList:
            node.captureNum = 0
            node.lifeTime += 1