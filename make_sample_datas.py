import numpy as np
from numpy import array, kron, trace, identity, sqrt, zeros, exp, pi, conjugate, random
from scipy.linalg import sqrtm
from datetime import datetime
from concurrent import futures
import os
import glob
from pathlib import Path


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

# baseVecter = np.zeros([1, 2**numberOfQubits])

# baseVecter = np.full([1, 2**numberOfQubits], 1/np.sqrt(2**numberOfQubits), dtype=np.complex) #all

# baseVecter[0][1] = 1 / 2 #only one qubit 1 |0001> + |0010> + |0100> + |1000>
# baseVecter[0][2] = 1 / 2
# baseVecter[0][4] = 1 / 2
# baseVecter[0][8] = 1 / 2

# baseVecter[0][0] = 1 / sqrt(2)                    #1111ed (|0000>+|1111>)(<0000|+<1111|)
# baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)

# matrix = baseVecter.T @ baseVecter

# matrix = np.zeros([2**numberOfQubits, 2**numberOfQubits]) # (|0000>+|1111>)(<0000|+<1111|) + |0001><0001| + |0010><0010| + |0100><0100| + |1000><1000|
# baseVecter = np.zeros([1, 2**numberOfQubits])
# baseVecter[0][1] = 1
# matrix += baseVecter.T @ baseVecter
# baseVecter = np.zeros([1, 2**numberOfQubits])
# baseVecter[0][2] = 1
# matrix += baseVecter.T @ baseVecter
# baseVecter = np.zeros([1, 2**numberOfQubits])
# baseVecter[0][4] = 1
# matrix += baseVecter.T @ baseVecter
# baseVecter = np.zeros([1, 2**numberOfQubits])
# baseVecter[0][8] = 1
# matrix += baseVecter.T @ baseVecter

# baseVecter = np.zeros([1, 2**numberOfQubits])
# baseVecter[0][0] = 1
# baseVecter[0][2**numberOfQubits-1] = 1
# matrix += baseVecter.T @ baseVecter

# matrix = matrix/np.trace(matrix)



# with open("./testdata/4qubitsdataramdom.txt", mode='a') as f:
#     for base in bases:
#         c = np.real(np.trace(matrix @ base) * 1000000)
#         f.writelines(str(int(c)) + " ")

if not os.path.exists('./testdata/4qubitsramdom'):
    os.makedirs('./testdata/4qubitsramdom')

for i in range(100):
    with open("./testdata/4qubitsramdom/"+str(i)+".txt", mode='a') as f:
        # for base in bases:
            # c = np.real(np.trace(matrix @ base) * 1000000)
            # f.writelines(str(int(random.poisson(c))) + " ")
        for _ in range((2**numberOfQubits)*(2**numberOfQubits)):
            f.writelines(str(random.randint(1, 100000)) + " ")
        f.writelines("\n")