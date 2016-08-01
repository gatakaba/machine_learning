# coding:utf-8
"""
フリーソフトで作る音声認識システム P118
系列データは {A,B}から生成されるとする
S1→S2→SE

遷移確率 P(zn|zn-1)
P(zn=S1|zn-1=S1)=0.8
P(zn=S2|zn-1=S1)=0.2
P(zn=SE|zn-1=S1)=0.0

P(zn=S1|zn-1=S2)=0.0
P(zn=S2|zn-1=S2)=0.6
P(zn=SE|zn-1=S2)=0.4

P(zn=S1|zn-1=S2)=0.0
P(zn=S2|zn-1=S2)=0.0
P(zn=SE|zn-1=S2)=1.0

出力確率 P(x|z)
P(x=A|z=S1)=0.8
P(x=B|z=S1)=0.2

P(x=A|z=S2)=0.4
P(x=B|z=S2)=0.6

P(x=A|z=SE)=0.5
P(x=B|z=SE)=0.5
"""
import numpy as np
import matplotlib.pyplot as plt

# 遷移行列
S = np.array([[0.8, 0.0, 0.0], [0.2, 0.6, 0.0], [0.0, 0.4, 1.0]])
# 条件付き観測確率
X = np.array([[0.8, 0.4, 0.5], [0.2, 0.6, 0.5]])
# パス[S1,S1,S2,SE]を通り,[A,B,A]を出力する確率
z = [0, 0, 1, 2]
x = [0, 1, 0, 0]

