
import numpy as np
import matplotlib.pyplot as plt

actionList = [[1, 0], [0, 1], [-1, 0], [0, -1] ]

def isWall(x,y):
    if x > 0:
        return True
    else:
        return False
    
x, y = 0, 0

tmpList = []
"""
for action in  actionList:
    dx, dy = action
    if not isWall(x + dx, y + dy):
        tmpList.append([dx, dy])
tmpList = []
"""

print [[dx, dy] for dx, dy in actionList if not isWall(x + dx, y + dy)]

