#coding:utf-8
#平滑化フィルタをかける
from matplotlib.cm import jet
from pandas.stats.moments import ewma

import matplotlib.pyplot as plt
import numpy as np


colors = jet(np.linspace(0,1,12))
def ToIEMG(EMG):
    #平滑化した後、パワーが一定になるように正規化
    return ewma(np.abs(EMG),span=150)/0.265 

def SMA(X,filterLength=127):
    #Simple Mean Average
    return ewma(X,span=filterLength) 

def InverseNormalize(x):
    #最大値が0,最小値が1になるように逆転した上で[0,1]の範囲に正規化を行う
    return -(x-np.min(x))/(np.max(x)-np.min(x))+1

def ToAIEMG(IEMG,t=0.2,samplingRate=1000):
    #現時点から過去t[s]の平均値を計算
    if len(IEMG.shape)==1:
        filterLength=t*samplingRate
        tmp=np.r_[np.ones(filterLength)*IEMG[0],IEMG]
        return np.convolve(tmp,np.ones(filterLength)/filterLength,"same")[:-filterLength]
    else:
        l=[]
        for i in range(IEMG.shape[1]):
            filterLength=t*samplingRate
            tmp=np.r_[np.ones(filterLength)*IEMG[0,i],IEMG[:,i]]
            l.append(np.convolve(tmp,np.ones(filterLength)/filterLength,"same")[:-filterLength])
        l=np.array(l).T
        return l
if __name__=="__main__":
    x=np.cumsum(np.random.normal(size=[10000,2]),axis=0)
    #x=np.cumsum(np.random.normal(size=10000),axis=0)
    
    z = ToAIEMG(x)
    plt.plot(x,label="Raw Wave")
    plt.plot(z,label="Averaged Wave")
    plt.legend()
    plt.show()
    
