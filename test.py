import numpy as np
from numpy import array, kron, trace, identity
from scipy.linalg import sqrtm, funm, inv

root2 = np.roots(2)

b0 = array([[1,0],[0,0]])
bp = array([[0.5,0.5],[0.5,0.5]])

def calculate():
    h = b0 + bp
    hi = inv(h)
    hir = sqrtm(hi)
    return hir@(bp + b0)@hir , h

if __name__ == "__main__":
    a, h = calculate()
    print(a)
    print(h)