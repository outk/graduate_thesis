'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
for quantum state tomography.

This is for multiple qubits system.

'''

import numpy as np
from numpy import array, kron, trace, identity, sqrt, random
import scipy
from scipy.linalg import sqrtm, funm
from datetime import datetime, timedelta


""" 
definition of base: 

    bH (numpy.array (2*2))
    bV (        "        )
    bD (        "        )
    bR (        "        )
    bL (        "        )

"""

bH = array([[1,0],[0,0]])
bV = array([[0,0],[0,1]])
bD = array([[1/2,1/2],[1/2,1/2]])
bR = array([[1/2,1j/2],[-1j/2,1/2]])
bL = array([[1/2,-1j/2],[1j/2,1/2]])



""" 
matrix of bases for multiple qubits: 
    
    bases (numpy.array ((2 ** [number of qubits] * 2 ** [number of qubits]) * (4 ** [number of qubits]))) 


------------------
A order of this bases array is one of most important things in calculation.
So you must match each other between this and data set.

"""


"""
Make Measurement Bases For Multiple Qubits

"""

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



"""
Get Experimental Datas

"""

def getDatasFromFile(fileOfExperimentalDatas, numberOfQubits):
    """
    getDatasFromFile(fileOfExperimentalDatas, numberOfQubits):

        This function is getting datas from file of experiment consequense,
        and return matrix (np.array (numberOfQubits*numberOfQubits)) of them.

    """
    matrixOfExperimentalDatas = np.zeros([numberOfQubits,numberOfQubits], dtype=np.complex)
    # TODO: modify matrixOfExperimentalDatas by given data file.
    return matrixOfExperimentalDatas



"""
Iterative Algorithm 

"""

def doIterativeAlgorithm(numberOfQubits, bases, maxNumberOfIteration, listOfExperimentalDatas):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) and diluted MLE algorithm(10.1103/PhysRevA.75.042108) to a set of datas given from a experiment.
        This recieve four variables (numberOfQubits, bases, maxNumberOfIteration, listAsExperimentalDatas),
        and return most likely estimated density matrix (np.array) and total time of calculation(datetime.timedelta).

        First quantum state matrix for this algorithm is a identity matrix.


        --------------------------------------------------------------------------------------------------------------
        Return:

            most likely estimated density matrix(np.array), 
            time difference(datetime.timedelta)

    """

    """ Setting initial parameters """
    iter = 0
    epsilon = 1000
    TolFun = 10e-11
    endDiff = 10e-10
    diff = 100
    traceDistance = 100

    dataList = listOfExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList / totalCountOfData # nDataList is a list of normarized datas

    densityMatrix = identity(2 ** numberOfQubits) # Input Density Matrix in Diluted MLE  (Identity)

    startTime = datetime.now() #Timestamp

    
    """ Start iteration """
    while traceDistance > TolFun and iter <= maxNumberOfIteration:
    # while diff > endDiff and iter <= maxNumberOfIteration:

        probList = [trace(bases[i] @ densityMatrix) for i in range(len(bases))]
        nProbList = probList / sum(probList)
        rotationMatrix = sum([(nDataList[i] / probList[i]) * bases[i] for i in range(4 ** numberOfQubits)])


        """ Normalization of Matrices for Measurement Bases """
        U = np.linalg.inv(sum(bases)) / sum(probList)   
        rotationMatrixLeft = (identity(2 ** numberOfQubits) + epsilon * U @ rotationMatrix) / (1 + epsilon)
        rotationMatrixRight = (identity(2 ** numberOfQubits) + epsilon * rotationMatrix @ U) / (1 + epsilon)


        """ Calculation of updated density matrix """
        modifiedDensityMatrix = rotationMatrixLeft @ densityMatrix @ rotationMatrixRight / trace(rotationMatrixLeft @ densityMatrix @ rotationMatrixRight)
        eigValueArray, eigVectors = np.linalg.eig(densityMatrix - modifiedDensityMatrix)
        traceDistance = sum(np.absolute(eigValueArray)) / 2


        """ Update Likelihood Function, and Compared with older one """
        LikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(4 ** numberOfQubits)])
        probList = [trace(bases[i] @ modifiedDensityMatrix) for i in range(len(bases))]
        nProbList = probList / sum(probList)
        modifiedLikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(4 ** numberOfQubits)])
        
        diff = modifiedLikelihoodFunction - LikelihoodFunction


        """ Show Progress of Calculation """
        progress = 100 * iter / maxNumberOfIteration
        if progress % 5 == 0:
            msg = "Progress of calculation: " + str(int(progress)) + "%"
            print(msg)


        """ Increment """
        iter += 1


        """ Check Increasing of Likelihood Function  """
        if diff < 0:
            epsilon = epsilon * 0.1
            continue
        

        """ Update Density Matrix """
        densityMatrix = modifiedDensityMatrix.copy()


    endTime = datetime.now() #Timestamp
    

    """ Check That Max Iteration Number was appropriate """
    if iter >= maxNumberOfIteration:
        print("----------------------------------------------")
        print("Iteration time reached max iteration number.")
        print("The number of max iteration times is too small.")
        print("----------------------------------------------")

    
    """ Show the total number of iteration """
    endIterationTimes = iter
    emsg = "Iteration was '" + str(endIterationTimes) + "' times."
    print(emsg)

    return modifiedDensityMatrix, endTime - startTime



"""
Calculate Fidelity

"""

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):



    """
    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix)) @ sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix))))

    return fidelity



if __name__ == "__main__":
    # listOfExperimentalDatas = array(list(map(float, input().split())))

    # listOfExperimentalDatas = array([5.0, 3335.3333333333334, 1665.6666666666667, 1665.6666666666667, 5.0, 6.0, 6.0, 7.0, 4.0, 1665.6666666666667, 835.33333333333336, 835.33333333333336, 835.33333333333336, 853.33333333333336, 1665.6666666666667, 2.0, 6665.6666666666669, 4.0, 3335.3333333333334, 3335.3333333333334, 1665.6666666666667, 1665.6666666666667, 3.0, 3335.3333333333334, 3335.3333333333334, 3.0, 1665.6666666666667, 1665.6666666666667, 1665.6666666666667, 1665.6666666666667, 2.0, 3335.3333333333334, 1665.6666666666667, 1665.6666666666667, 3335.3333333333334, 1665.6666666666667, 835.33333333333336, 835.33333333333336, 5.0, 1665.6666666666667, 1665.6666666666667, 835.33333333333336, 2085.3333333333334, 415.66666666666668, 2085.33333333333337, 3755.0000000000001, 835.33333333333336, 3335.3333333333334, 3335.3333333333334, 853.33333333333336, 2085.33333333333337, 3755.0000000000001, 2085.3333333333334, 2085.33333333333337, 835.33333333333336, 1665.6666666666667, 1665.6666666666667, 4.0, 835.33333333333336, 835.33333333333336, 3335.3333333333334, 1665.6666666666667, 1665.6666666666667, 1665.6666666666667])

    listOfExperimentalDatas = 2 + array([499, 0, 249, 249, 0, 0, 0, 0, 249, 0, 124, 124, 124, 124, 0, 249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 0, 124, 124, 0, 0, 0, 0, 124, 0, 62, 62, 62, 62, 0, 124, 124, 0, 62, 62, 62, 62, 0, 124, 0, 0, 0, 0, 124, 124, 0, 249, 0, 0, 0, 0, 124, 124, 249, 0, 0, 124, 62, 62, 62, 62, 124, 0, 0, 124, 62, 62, 62, 62, 124, 0, 0, 249, 124, 124, 0, 0, 0, 0, 0, 0, 0, 0, 249, 249, 499, 0, 0, 249, 124, 124, 124, 124, 249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 0, 124, 124, 0, 0, 0, 0, 124, 0, 62, 62, 62, 62, 0, 124, 0, 124, 62, 62, 62, 62, 124, 0, 0, 249, 124, 124, 0, 0, 0, 0, 124, 0, 62, 62, 62, 62, 124, 0, 62, 62, 124, 62, 0, 62, 62, 62, 62, 62, 0, 62, 0, 62, 62, 62, 0, 124, 62, 62, 62, 62, 0, 124, 124, 0, 62, 62, 62, 62, 124, 0, 62, 62, 0, 62, 124, 62, 62, 62, 62, 62, 0, 62, 0, 62, 62, 62, 0, 124, 62, 62, 62, 62, 0, 124, 0, 0, 0, 0, 124, 124, 249, 0, 0, 124, 62, 62, 62, 62, 124, 0, 124, 0, 62, 62, 62, 62, 0, 124, 0, 0, 0, 0, 124, 124, 0, 249])

    maxNumberOfIteration = 10000000

    numberOfQubits = 4

    bases = makeBases(numberOfQubits)

    estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(numberOfQubits, bases, maxNumberOfIteration, listOfExperimentalDatas)


    """ simulation of ghz state """
    # ghz = np.zeros([8,8])
    # ghz[0][0] = 1 / 2
    # ghz[7][7] = 1 / 2
    # ghz[0][7] = 1 / 2
    # ghz[7][0] = 1 / 2
    # fidelity = calculateFidelity(ghz, estimatedDensityMatrix)


    """ simulation of w state """
    # w = np.zeros([1,8])
    # w[0][4] = 1 / sqrt(3)
    # w[0][6] = 1 / sqrt(3)
    # w[0][1] = 1 / sqrt(3)
    # wmatrix = w.T @ w
    # fidelity = calculateFidelity(wmatrix, estimatedDensityMatrix)

    """ example of four qubits """
    baseVecter = np.zeros([1, 2**numberOfQubits])
    baseVecter[0][0] = 1 / sqrt(2)
    baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)
    matrix = baseVecter.T @ baseVecter
    fidelity = calculateFidelity(matrix, estimatedDensityMatrix)

    print(estimatedDensityMatrix)

    print("Fidelity is " + str(fidelity))

    print("Time of calculation: ", timeDifference)


    # ghz = np.zeros([8,8])

    # ghz[0][0] = 1 / 2
    # ghz[7][7] = 1 / 2
    # ghz[0][7] = 1 / 2
    # ghz[7][0] = 1 / 2

    # w = np.zeros([1,8])

    # w[0][4] = 1 / sqrt(3)
    # w[0][6] = 1 / sqrt(3)
    # w[0][1] = 1 / sqrt(3)

    # wmatrix = w.T @ w

    # print(wmatrix)

    # baseVecter = np.zeros([1, 2**numberOfQubits])
    # baseVecter[0][0] = 1 / sqrt(2)
    # baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)

    # matrix = baseVecter.T @ baseVecter

    # for base in bases:
    #     ls.append(int(np.real(trace(matrix @ base)) * 1000))
        
    
    # print(ls)


