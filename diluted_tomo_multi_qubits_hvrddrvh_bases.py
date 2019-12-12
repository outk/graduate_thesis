'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
for quantum state tomography.

This is for multiple qubits system.

'''

import numpy as np
from numpy import array, kron, trace, identity, sqrt
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

    listOfExperimentalDatas = array([0.0, 333.3333333333334, 166.6666666666667, 166.6666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 166.6666666666667, 83.33333333333336, 83.33333333333336, 83.33333333333336, 83.33333333333336, 166.6666666666667, 0.0, 666.6666666666669, 0.0, 333.3333333333334, 333.3333333333334, 166.6666666666667, 166.6666666666667, 0.0, 333.3333333333334, 333.3333333333334, 0.0, 166.6666666666667, 166.6666666666667, 166.6666666666667, 166.6666666666667, 0.0, 333.3333333333334, 166.6666666666667, 166.6666666666667, 333.3333333333334, 166.6666666666667, 83.33333333333336, 83.33333333333336, 0.0, 166.6666666666667, 166.6666666666667, 83.33333333333336, 208.3333333333334, 41.66666666666668, 208.33333333333337, 375.0000000000001, 83.33333333333336, 333.3333333333334, 333.3333333333334, 83.33333333333336, 208.33333333333337, 375.0000000000001, 208.3333333333334, 208.33333333333337, 83.33333333333336, 166.6666666666667, 166.6666666666667, 0.0, 83.33333333333336, 83.33333333333336, 333.3333333333334, 166.6666666666667, 166.6666666666667, 166.6666666666667])

    maxNumberOfIteration = 10000000

    numberOfQubits = 3

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

    print(estimatedDensityMatrix)

    # print("Fidelity is " + str(fidelity))

    print("Time of calculation: ", timeDifference)

    # ls = []

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


    # for base in bases:
    #     ls.append(np.real(trace(wmatrix @ base)) * 1000)
    
    # print(ls)


