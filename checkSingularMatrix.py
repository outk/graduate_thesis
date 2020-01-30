import numpy as np
from numpy import array, kron, trace, identity, sqrt, random
import scipy
from scipy.linalg import sqrtm, funm, eig
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
su2Bases = array(su2Bases) / 8


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

    # print(B)

    BInverse = np.linalg.inv(B)

    M = []

    for i in range(4**numberOfQubits):
        M.append(sum([BInverse[j][i] * su2Bases[j] for j in range(4**numberOfQubits)]))
    
    return array(M)


def makeDensityMatrix(numberOfQubits, dataList, bases):
    M = makeMMatrix(numberOfQubits, bases)

    # print(M)

    N = sum([np.trace(M[i]) * dataList[i] for i in range(4**numberOfQubits)])

    densityMatrix = sum([dataList[i] * M[i] for i in range(4**numberOfQubits)]) / N

    # print(densityMatrix)

    # print(trace(densityMatrix))

    # print(trace(densityMatrix @ densityMatrix))
    # print(eig(densityMatrix))

    return densityMatrix


""" cholesky decomposition """
def choleskyDecomposition(numberOfQubits, matrix):

    L = np.zeros([2**numberOfQubits, 2**numberOfQubits], dtype=np.complex)

    for i in range(2**numberOfQubits-1, -1, -1):
        s = matrix[i][i]
        for k in range(2**numberOfQubits-1, i, -1):
            s -= np.conjugate(L[k][i]) * L[k][i]
        if s != 0:
            L[i][i] = np.sqrt(s)
        else:
            L[i][i] = 0
        for j in range(i):
            t = matrix[i][j]
            for k in range(2**numberOfQubits-1, i, -1):
                t -= (np.conjugate(L[k][i]) * L[k][j])
            if L[i][i] != 0:
                L[i][j] = t / np.conjugate(L[i][i])
            else:
                L[i][j] = t / 1e-9

    for i in range(2**numberOfQubits):
        L[i][i] = np.real(L[i][i])

    return (np.conjugate(L).T @ L) / np.trace(np.conjugate(L).T @ L)

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):



    """
    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix))))
    return fidelity


if __name__ == "__main__":
    with open("./testdata/4qubitspoissondata/8.txt") as f:
        listOfExperimentalDatas = []
        for s in f.readlines():
            listOfExperimentalDatas.extend(map(int, s.strip().split()))

    numberOfQubits = 4

    bases = makeBases(numberOfQubits)

    densityMatrix = makeDensityMatrix(numberOfQubits, listOfExperimentalDatas, bases)

    initialDensityMatrix = choleskyDecomposition(numberOfQubits, densityMatrix)

    # l = np.linalg.cholesky(densityMatrix)

    # print(densityMatrix - np.conjugate(l).T @ l)

    # print(densityMatrix - initialDensityMatrix)

    # print(np.trace(initialDensityMatrix))

    print(eig(initialDensityMatrix))

    print(eig(densityMatrix))

    # print(initialDensityMatrix)

    # print(np.matrix(initialDensityMatrix).getH())

    # print(np.matrix(initialDensityMatrix).getH()-initialDensityMatrix)

    # print(np.matrix(densityMatrix).getH()-densityMatrix)


    baseVecter = np.zeros([1, 2**numberOfQubits])
    # baseVecter[0][0] = 1 / sqrt(2)
    # baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)
    baseVecter[0][1] = 1 / 2
    baseVecter[0][2] = 1 / 2
    baseVecter[0][4] = 1 / 2
    baseVecter[0][8] = 1 / 2
    idealDensityMatrix = baseVecter.T @ baseVecter
    matrix = baseVecter.T @ baseVecter
    fidelity = calculateFidelity(matrix, initialDensityMatrix)

    print(fidelity)

    # fidelity = calculateFidelity(matrix, densityMatrix)

    # print(fidelity)

