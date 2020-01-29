import numpy as np
from numpy import array, kron, trace, identity, sqrt, random
import scipy
from scipy.linalg import sqrtm, funm
from datetime import datetime, timedelta

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

def makeBMatrix(numberOfQubits, bases):
    B = np.zeros((4**numberOfQubits, 4**numberOfQubits))

    for i in range(4**numberOfQubits):
        for j in range(4**numberOfQubits):
            B[i][j] = np.trace(bases[i] @ su2Bases[j])

    return B


def makeMMatrix(numberOfQubits, bases):
    B = makeBMatrix(numberOfQubits, bases)

    BInverse = np.linalg.inv(B)

    M = []

    for i in range(4**numberOfQubits):
        M.append(sum([BInverse[j][i] * su2Bases[j] for j in range(4**numberOfQubits)]))
    
    return array(M)


def makeDensityMatrix(numberOfQubits, dataList, bases):
    M = makeMMatrix(numberOfQubits, bases)

    print(M)

    N = sum([np.trace(M[i]) * dataList[i] for i in range(4**numberOfQubits)])

    print(N)

    print(dataList)

    densityMatrix = sum([dataList[i] * M[i] for i in range(4**numberOfQubits)]) / N

    print(densityMatrix)

    return densityMatrix


""" cholesky decomposition """
def choleskyDecomposition(numberOfQubits, matrix):

    L = np.zeros([2**numberOfQubits, 2**numberOfQubits], dtype=np.complex)

    for i in range(2**numberOfQubits):
        for j in range(i-1):
            s = matrix[i][j]
            for k in range(j-1):
                s -= np.conjugate(L[i][k]) * L[j][k]
            if L[j][j] != 0:
                L[i][j] = s / L[j][j]
            else:
                L[i][j] = s / 1e-9
        s = matrix[i][i]
        for k in range(i-1):
            s -= np.conjugate(L[i][k])*L[i][k]
        if np.real(s) < 0:
            s = -1*s
        L[i][i] = np.sqrt(s)

    return np.conjugate(L).T @ L / np.trace(np.conjugate(L).T @ L)

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):



    """
    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix)))) ** 2
    return fidelity


if __name__ == "__main__":
    with open("./testdata/4qubitspoissondata/6.txt") as f:
        listOfExperimentalDatas = []
        for s in f.readlines():
            listOfExperimentalDatas.extend(map(int, s.strip().split()))

    numberOfQubits = 4

    bases = makeBases(numberOfQubits)

    densityMatrix = makeDensityMatrix(numberOfQubits, listOfExperimentalDatas, bases)

    initialDensityMatrix = choleskyDecomposition(numberOfQubits, densityMatrix)

    print(np.trace(initialDensityMatrix))

    baseVecter = np.zeros([1, 2**numberOfQubits])
    # baseVecter[0][0] = 1 / sqrt(2)
    # baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)
    baseVecter[0][1] = 1 / 2
    baseVecter[0][2] = 1 / 2
    baseVecter[0][4] = 1 / 2
    baseVecter[0][8] = 1 / 2
    idealDensityMatrix = baseVecter.T @ baseVecter
    matrix = baseVecter.T @ baseVecter
    fidelity = calculateFidelity(matrix, densityMatrix)

    print(fidelity)

    fidelity = calculateFidelity(matrix, identity(2**numberOfQubits) / 2**numberOfQubits)

    print(fidelity)

