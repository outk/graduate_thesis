import numpy as np
from numpy import array, kron, trace, identity, sqrt, zeros, exp, pi, conjugate, random
from scipy.linalg import sqrtm
from datetime import datetime
from concurrent import futures
import os


su2b = array([
    [[   1,  0], [   0,  1]],
    [[   0,  1], [   1,  0]],
    [[   0,-1j], [  1j,  0]],
    [[   1,  0], [   0, -1]]
]
)
su2Bases = []
newbases = []
for i in range(4):
    su2Bases.extend([kron(su2b[i], su2b[j]) for j in range(4)])
newbases = su2Bases.copy()
su2Bases = []
for i in range(16):
    su2Bases.extend([kron(newbases[i], su2b[j]) for j in range(4)])
newbases = su2Bases.copy()
su2Bases = []
for i in range(64):
    su2Bases.extend([kron(newbases[i], su2b[j]) for j in range(4)])
su2Bases = array(su2Bases)


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

numberOfQubits = 4

bases = makeBases(numberOfQubits)

baseVecter = np.zeros([1, 2**numberOfQubits])

baseVecter[0][1] = 1 / 2
baseVecter[0][2] = 1 / 2
baseVecter[0][4] = 1 / 2
baseVecter[0][8] = 1 / 2
matrix = baseVecter.T @ baseVecter

with open("./testdata/4qubitsdata.txt", mode='a') as f:
    for base in bases:
        c = np.real(np.trace(matrix @ base) * 1000000)
        f.writelines(str(int(c)) + " ")

for i in range(1000):
    with open("./testdata/4qubitspoissondata/"+str(i)+".txt", mode='a') as f:
        for base in bases:
            c = np.real(np.trace(matrix @ base) * 1000000)
            f.writelines(str(int(random.poisson(c))) + " ")
        f.writelines("\n")