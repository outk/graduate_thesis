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

    BInverse = np.linalg.inv(B)

    M = []

    for i in range(4**numberOfQubits):
        M.append(sum([BInverse[j][i] * su2Bases[j] for j in range(4**numberOfQubits)]))
    
    return array(M)


def makeInitialDensityMatrix(numberOfQubits, dataList, bases):
    M = makeMMatrix(numberOfQubits, bases)

    N = sum([np.trace(M[i]) * dataList[i] for i in range(4**numberOfQubits)])

    densityMatrix = sum([dataList[i] * M[i] for i in range(4**numberOfQubits)]) / N

    initialDensityMatrix = choleskyDecomposition(numberOfQubits, densityMatrix)

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
                L[i][j] = t / 1e-9

    for i in range(2**numberOfQubits):
        L[i][i] = np.real(L[i][i])

    return (np.conjugate(L).T @ L) / np.trace(np.conjugate(L).T @ L)


""" Get Experimental Data """

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



def doIterativeAlgorithm(numberOfQubits, bases, listOfExperimentalDatas, idealDensityMatrix):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) and diluted MLE algorithm(10.1103/PhysRevA.75.042108) to a set of datas given from a experiment.
        This recieve four variables (numberOfQubits, bases, listAsExperimentalDatas),
        and return most likely estimated density matrix (np.array) and total time of calculation(datetime.timedelta).

        First quantum state matrix for this algorithm is a identity matrix.


        --------------------------------------------------------------------------------------------------------------
        Return:

            most likely estimated density matrix(np.array), 

    """

    """ Setting initial parameters """
    iter = 0
    epsilon = 1000
    endDiff = 1e-11
    diff = 100
    # TolFun = 10e-11
    # traceDistance = 100
    maxNumberOfIteration = 100000

    dataList = listOfExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList / totalCountOfData # nDataList is a list of normarized datas
    densityMatrix = makeInitialDensityMatrix(numberOfQubits, dataList, bases)
    # densityMatrix = identity(2 ** numberOfQubits)

    fidelity = calculateFidelity(idealDensityMatrix, densityMatrix)
    with open('fpusedmoqall.txt', mode='a') as f:
        f.writelines(str(fidelity) + '\n')

    """ Start iteration """
    # while traceDistance > TolFun and iter <= maxNumberOfIteration:
    while diff > endDiff and iter <= maxNumberOfIteration and epsilon > 1e-6:

        probList = [trace(base @ densityMatrix) for base in bases]
        nProbList = probList / sum(probList)
        rotationMatrix = sum([(nDataList[i] / probList[i]) * bases[i] for i in range(4 ** numberOfQubits) if probList[i] != 0])

        """ Normalization of Matrices for Measurement Bases """
        U = np.linalg.inv(sum(bases)) / sum(probList)   
        rotationMatrixLeft = (identity(2 ** numberOfQubits) + epsilon * U @ rotationMatrix) / (1 + epsilon)
        rotationMatrixRight = (identity(2 ** numberOfQubits) + epsilon * rotationMatrix @ U) / (1 + epsilon)

        """ Calculation of updated density matrix """
        modifiedDensityMatrix = rotationMatrixLeft @ densityMatrix @ rotationMatrixRight / trace(rotationMatrixLeft @ densityMatrix @ rotationMatrixRight)
        eigValueArray, eigVectors = np.linalg.eig(densityMatrix - modifiedDensityMatrix)
        traceDistance = sum(np.absolute(eigValueArray)) / 2

        """ Update Likelihood Function, and Compared with older one """
        LikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(4 ** numberOfQubits) if nProbList[i] != 0])
        probList = [trace(base @ modifiedDensityMatrix) for base in bases]
        nProbList = probList / sum(probList)
        modifiedLikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(4 ** numberOfQubits) if nProbList[i] != 0])
        nowdiff = np.real(modifiedLikelihoodFunction - LikelihoodFunction)

        """ Show Progress of Calculation """
        progress = 100 * iter / maxNumberOfIteration
        if progress % 5 == 0:
            msg = "Progress of calculation: " + str(int(progress)) + "%"
            print(msg)

        """ Increment """
        iter += 1

        """ Check Increasing of Likelihood Function  """
        if nowdiff <= 0:
            epsilon = epsilon * 0.1
            print("failed")
            continue
        else:
            diff = nowdiff
        
        """ Update Density Matrix """
        densityMatrix = modifiedDensityMatrix.copy()

        fidelity = calculateFidelity(idealDensityMatrix, densityMatrix)
        with open('fpusedmoqall.txt', mode='a') as f:
            f.writelines(str(fidelity) + ' ' + str(epsilon) + ' ' + str(nowdiff) + '\n')

    print(densityMatrix)



    """ Check That Max Iteration Number was appropriate """
    if iter >= maxNumberOfIteration:
        print("----------------------------------------------")
        print("Iteration time reached max iteration number.")
        print("The number of max iteration times is too small.")
        print("----------------------------------------------")

    """ Show the total number of iteration """
    endIterationTimes = iter
    print("Iteration was '" + str(endIterationTimes) + "' times.")

    return modifiedDensityMatrix, endIterationTimes



""" Calculate Fidelity """

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):

    """

    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix)))) ** 2

    return fidelity



""" Get Number of Qubits """

def getNumberOfQubits():
    """
    getNumberOfQubits()

    """

    print("------------------------------------------------------------")
    print("PLEASE ENTER NUMBER OF QUBITS")
    print("------------------------------------------------------------")
    print(">>")

    numberOfQubits = int(input())

    return numberOfQubits



if __name__ == "__main__":

    """ Get Number of Qubits """
    numberOfQubits = getNumberOfQubits()

    # """ Make SU2 Bases """
    # su2Bases = makeSU2Bases(numberOfQubits)
    
    """ Get Path of Experimental Data Directory """
    print("input data path ")
    path = input()

    """ Make Bases """
    basesOfQubits = makeBases(numberOfQubits)

    """ Make Ideal Density Matrix """
    # baseVecter = np.zeros([1, 2**numberOfQubits])
    # baseVecter[0][0] = 1 / sqrt(2)
    # baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)
    # baseVecter[0][1] = 1 / 2
    # baseVecter[0][2] = 1 / 2
    # baseVecter[0][4] = 1 / 2
    # baseVecter[0][8] = 1 / 2

    # baseVecter = np.full([1, 2**numberOfQubits], 1/np.sqrt(2**numberOfQubits), dtype=np.complex)
    # idealDensityMatrix = baseVecter.T @ baseVecter

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
    # idealDensityMatrix = matrix

    baseVecter = np.full([1, 2**numberOfQubits], 1/np.sqrt(2**numberOfQubits), dtype=np.complex)
    idealDensityMatrix = baseVecter.T @ baseVecter

    start_time = datetime.now() #time stamp

    
    listOfExperimentalData = getExperimentalData(path)
    m, it = doIterativeAlgorithm(numberOfQubits, basesOfQubits, listOfExperimentalData, idealDensityMatrix)

    end_time = datetime.now() #time stamp
    print("Total Calculation Time was " + str(end_time - start_time))


    # with open('benchmark'+str(numberOfQubits)+'qubits.txt', mode='a') as f:
    #     f.write("total time: " + str(end_time - start_time) + "\n")

