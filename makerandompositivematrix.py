'''
This is a implementation of Quantum State Tomography for Qubits,
using techniques of following papars.

'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
'Qudit Quantum State Tomography(10.1103/PhysRevA.66.012303)'

'''

import numpy as np
from numpy import array, kron, trace, identity, sqrt, zeros, exp, pi, conjugate, random
from scipy.linalg import sqrtm
from datetime import datetime
from concurrent import futures
import os
import glob
from pathlib import Path


# su2b = array([
#     [[   1,  0], [   0,  1]],
#     [[   0,  1], [   1,  0]],
#     [[   0,-1j], [  1j,  0]],
#     [[   1,  0], [   0, -1]]
# ]
# )

# def makeSU2Bases(numberOfQubits):
#     newbases = su2b.copy()
#     su2Bases = []
#     for i in range(numberOfQubits-1):
#         for newbase in newbases:
#             su2Bases.extend([kron(newbase, base) for base in su2b])
#         newbases = su2Bases.copy()
#         su2Bases = []

#     return array(su2Bases)

su2b = array([
    [[   1,  0], [   0,  1]],
    [[   0,  1], [   1,  0]],
    [[   0,-1j], [  1j,  0]],
    [[   1,  0], [   0, -1]]
]
)

su2Bases = []
newbases = su2b.copy()

def makeSU2Bases(numberOfQubits):
    global newbases, su2Bases
    for _ in range(numberOfQubits-1):
        for i in range(len(newbases)):
            su2Bases.extend([kron(newbases[i], su2b[j]) for j in range(4)])
        newbases = su2Bases.copy()
        su2Bases = []
        
    su2Bases = array(newbases) / (2**numberOfQubits)


bH = array([[1,0],[0,0]])
bV = array([[0,0],[0,1]])
bD = array([[1/2,1/2],[1/2,1/2]])
bR = array([[1/2,1j/2],[-1j/2,1/2]])
bL = array([[1/2,-1j/2],[1j/2,1/2]])

initialBases = array([bH, bV, bR, bD])

cycleBases1 = array([bH, bV, bR, bD])
cycleBases2 = array([bD, bR, bV, bH])

def makeBases(numberOfQubits):
    beforeBases = initialBases
    for _ in range(numberOfQubits - 1):
        afterBases = []
        for i in range(len(beforeBases)):
            if i % 2 == 0:
                afterBases.extend([kron(beforeBases[i], cycleBase) for cycleBase in cycleBases1])
            else:
                afterBases.extend([kron(beforeBases[i], cycleBase) for cycleBase in cycleBases2])
        beforeBases = afterBases
    return array(afterBases)

def makeBMatrix(numberOfQubits, bases):
    global su2Bases
    B = np.zeros((4**numberOfQubits, 4**numberOfQubits))

    for i in range(4**numberOfQubits):
        for j in range(4**numberOfQubits):
            B[i][j] = np.trace(bases[i] @ su2Bases[j])

    return B


def makeMMatrix(numberOfQubits, bases):
    global su2Bases
    B = makeBMatrix(numberOfQubits, bases)

    BInverse = np.linalg.inv(B)

    M = []

    for i in range(4**numberOfQubits):
        M.append(sum([BInverse[j][i] * su2Bases[j] for j in range(4**numberOfQubits)]))
    
    return array(M)


def makeInitialDensityMatrix(numberOfQubits, dataList, bases, M):
    # M = makeMMatrix(numberOfQubits, bases)

    N = sum([np.trace(M[i]) * dataList[i] for i in range(4**numberOfQubits)])

    densityMatrix = sum([dataList[i] * M[i] for i in range(4**numberOfQubits)]) / N

    return np.all(np.real(np.linalg.eigvals(densityMatrix)) >= -1e-10)
    

if __name__ == "__main__":
    numberOfQubits = 4
    
    if not os.path.exists('./testdata/4qubitspositiverandom'):
        os.makedirs('./testdata/4qubitspositiverandom')

    makeSU2Bases(numberOfQubits)

    bases = makeBases(numberOfQubits)

    mmatrix = makeMMatrix(numberOfQubits, bases)

    iter = 0

    c = 0

    while iter < 100:
        print(c)
        datalist = [random.randint(1, 10000) for _ in range((2**numberOfQubits)*(2**numberOfQubits))]
        # datalist = []
        # with open('./testdata/4qubitsdata.txt') as f:
        #     s = f.readlines()
        #     for ss in s:
        #         datalist.extend(map(int, ss.strip().split()))
        
        if makeInitialDensityMatrix(numberOfQubits, datalist, bases, mmatrix):
            with open("./testdata/4qubitspositiverandom/"+str(iter)+".txt", mode='a') as f:
                f.writelines(str(datalist) + "\n")
            iter += 1
        c += 1