import numpy as np
from numpy import trace
from scipy import sqrt

ls = []

ghz = np.zeros([8,8])

ghz[0][0] = 1 / 2
ghz[7][7] = 1 / 2
ghz[0][7] = 1 / 2
ghz[7][0] = 1 / 2

w = np.zeros([1,8])

w[0][4] = 1 / sqrt(3)
w[0][6] = 1 / sqrt(3)
w[0][1] = 1 / sqrt(3)

wmatrix = w.T @ w

print(wmatrix)


for base in bases:
    ls.append(np.real(trace(wmatrix @ base)) * 1000)

print(ls)