import numpy as np
from numpy import array, sqrt, zeros, pi, exp, conjugate, kron, trace
from scipy.linalg import sqrtm

zero_base_array1 = zeros((1,3))
zero_base_array1[0][0] = 1
fb1 = zero_base_array1

zero_base_array2 = zeros((1,3))
zero_base_array2[0][1] = 1
fb2 = zero_base_array2

zero_base_array3 = zeros((1,3))
zero_base_array3[0][2] = 1
fb3 = zero_base_array3

mb1 = (conjugate((fb1 + fb2).T) @ (fb1 + fb2)) / 2
mb2 = (conjugate((fb1 + fb3).T) @ (fb1 + fb3)) / 2
mb3 = (conjugate((fb2 + fb3).T) @ (fb2 + fb3)) / 2
mb4 = (conjugate((exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3)) * fb2).T) @ (exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3) * fb2))) / 2
mb5 = (conjugate((exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3)) * fb2).T) @ (exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3) * fb2))) / 2
mb6 = (conjugate((exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3)) * fb3).T) @ (exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3) * fb3))) / 2
mb7 = (conjugate((exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3)) * fb3).T) @ (exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3) * fb3))) / 2
mb8 = (conjugate((exp( 2*pi*1j/3) * fb2 + (exp(-2*pi*1j/3)) * fb3).T) @ (exp( 2*pi*1j/3) * fb2 + (exp(-2*pi*1j/3) * fb3))) / 2
mb9 = (conjugate((exp(-2*pi*1j/3) * fb2 + (exp( 2*pi*1j/3)) * fb3).T) @ (exp(-2*pi*1j/3) * fb2 + (exp( 2*pi*1j/3) * fb3))) / 2

Bases = array([mb1, mb2, mb3, mb4, mb5, mb6, mb7, mb8, mb9])

su3b = np.array([
    [[   1,  0,  0], [   0,  1,  0], [   0,  0,  1]],
    [[   0,  1,  0], [   1,  0,  0], [   0,  0,  0]],
    [[   0,-1j,  0], [  1j,  0,  0], [   0,  0,  0]],
    [[   1,  0,  0], [   0, -1,  0], [   0,  0,  0]],
    [[   0,  0,  1], [   0,  0,  0], [   1,  0,  0]],
    [[   0,  0,-1j], [   0,  0,  0], [ -1j,  0,  0]],
    [[   0,  0,  0], [   0,  0,  1], [   0,  1,  0]],
    [[   0,  0,  0], [   0,  0,-1j], [   0,-1j,  0]],
    [[   1,  0,  0], [   0,  1,  0], [   0,  0, -2]] / sqrt(3),
])

su3Bases = []
for su3b1 in su3b:
    su3Bases.extend([kron(su3b1, su3b2) for su3b2 in su3b])
su3Bases = np.array(su3Bases)


def makeBases(numberOfQutrits, bases):
    for _ in range(numberOfQutrits-1):
        fixedBases = []
        for base1 in Bases:
            fixedBases.extend([kron(base1, base2) for base2 in Bases])
        bases = fixedBases.copy()

    return array(bases)

def makeBMatrix(numberOfQutrits, bases):
    B = np.zeros((9**numberOfQutrits, 9**numberOfQutrits))

    for i in range(9**numberOfQutrits):
        for j in range(9**numberOfQutrits):
            B[i][j] = np.trace(bases[i] @ su3Bases[j])

    print(B)

    return B


def makeRList(numberOfQutrits, dataList, bases):
    B = makeBMatrix(numberOfQutrits, bases)

    BInverse = np.linalg.inv(B)

    rlist = np.array([sum([BInverse[i][j] * dataList[j] for j in range(9**numberOfQutrits)]) for i in range(9**numberOfQutrits)])

    return rlist

def makeDensityMatrix(numberOfQutrits, dataList, bases):
    rList = makeRList(numberOfQutrits, dataList, bases)

    densityMatrix = sum([rList[i] * su3Bases[i] for i in range(9**numberOfQutrits)])
    densityMatrix = densityMatrix / np.trace(densityMatrix)

    return densityMatrix

def getExperimentalData(pathOfExperimentalData):
    """
    getExperimentalData(pathOfExperimentalData(string)):

        This function is getting experimental data from file at "pathOfExperimentalData",
        
        ----------------------------------------------------------------------------------------
        return:

            np.array of them.

    """

    with open(pathOfExperimentalData) as f:
        experimentalData = []
        for s in f.readlines():
            experimentalData.extend(map(int, s.strip().split()))

    return array(experimentalData)

def makeInitialDensityMatrix(numberOfQutrits, dataList, bases):

    # densityMatrix = np.sum(dataList*bases, 2)/np.sum(dataList*np.trace(bases))

    # densityMatrix = sum([dataList[i]*bases[i] for i in range(9 ** numberOfQutrits)]) / sum([dataList[i]*trace(bases[i]) for i in range(9**numberOfQutrits)])

    densityMatrix = makeDensityMatrix(numberOfQutrits, dataList, bases)

    print(densityMatrix)

    baseVecter = np.zeros([1, 3**numberOfQutrits])
    baseVecter[0][0] = 1 / sqrt(3)
    baseVecter[0][4] = 1 / sqrt(3)
    baseVecter[0][3**numberOfQutrits-1] = 1 / sqrt(3)
    idealDensityMatrix = baseVecter.T @ baseVecter

    print(idealDensityMatrix)

    fidelity = calculateFidelity(idealDensityMatrix, densityMatrix)

    print(fidelity)

    initialDensityMatrix = choleskyDecomposition(numberOfQutrits, densityMatrix)

    return initialDensityMatrix


    """ cholesky decomposition """
def choleskyDecomposition(numberOfQubits, matrix):

    L = np.zeros([2**numberOfQubits, 2**numberOfQubits], dtype=np.complex)

    for i in range(2**numberOfQubits-1, -1, -1):
        s = matrix[i][i]
        for k in range(2**numberOfQubits-1, i, -1):
            s -= np.conjugate(L[k][i]) * L[k][i]
        if s >= 0:
            L[i][i] = np.sqrt(s)
        else:
            L[i][i] = np.sqrt(s)
        for j in range(i):
            t = matrix[i][j]
            for k in range(2**numberOfQubits-1, i, -1):
                t -= (np.conjugate(L[k][i]) * L[k][j])
            if L[i][i] != 0:
                L[i][j] = np.conjugate(t / L[i][i])
            else:
                L[i][j] = np.conjugate(t / 1e-9)

    for i in range(2**numberOfQubits):
        L[i][i] = np.real(L[i][i])

    return (np.conjugate(L).T @ L) / np.trace(np.conjugate(L).T @ L)

    

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):

    """

    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix)) @ sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix))))

    return fidelity


if __name__ == "__main__":
    
    numberOfQutrits = 2

    newbases = makeBases(2, Bases)

    experimentalDataList = getExperimentalData("./testdata/data.txt")

    print(experimentalDataList)

    initialDensityMatrix = makeInitialDensityMatrix(numberOfQutrits, experimentalDataList, newbases)

    print(initialDensityMatrix)

    baseVecter = np.zeros([1, 3**numberOfQutrits])
    baseVecter[0][0] = 1 / sqrt(3)
    baseVecter[0][4] = 1 / sqrt(3)
    baseVecter[0][3**numberOfQutrits-1] = 1 / sqrt(3)
    idealDensityMatrix = baseVecter.T @ baseVecter

    fidelity = calculateFidelity(idealDensityMatrix, initialDensityMatrix)

    print(fidelity)

    