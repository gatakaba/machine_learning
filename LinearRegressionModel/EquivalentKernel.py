import numpy as np
import matplotlib.pyplot as plt
from Kernel import Kernel

N = 100
x = np.linspace(-0.5, 0.5, N)

alpha = 0.01
beta = 0.1
M = 10
kernel = Kernel(M)
phi = kernel.gauss(x)

S = np.linalg.inv(alpha * np.eye(N) + beta * np.dot(phi.T, phi))
print
S.shape
# K = np.dot(phi.T, S)

plt.pcolor(S)
plt.show()
