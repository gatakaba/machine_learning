# coding:utf-8
"""
非単調出力関数
この関数の本質は内部電位uの増加に対して
出力が単調に増加するのではなく,ある程度増加すると|f(u)|が減少する所にある
"""
import numpy as np
import matplotlib.pyplot as plt

def func(u):
    k = -1
    c = 50
    c_dash = 10
    h = 0.5
    
    tmp1 = (1 - np.exp(-c * u)) / (1 + np.exp(-c * u))
    tmp2 = (1 + k * np.exp(c_dash * (np.abs(u) - h))) / (1 + np.exp(c_dash * (np.abs(u) - h)))
    plt.plot(tmp1, linewidth = 3, alpha = 0.5)
    plt.plot(tmp2, linewidth = 3, alpha = 0.5)
    plt.plot(tmp1 * tmp2, linewidth = 3)
    


x = np.linspace(-5, 5, 1000)
y = func(x)

plt.show()