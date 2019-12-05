'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
for quantum state tomography.

This is for two qubits system.

'''

import numpy as np
from numpy import array, kron, trace, identity
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
matrix of bases for two qubits: 
    
    bases (numpy.array ((4*4)*16)) 


------------------
A order of this bases array is one of most important things in calculation.
So you must match each other between this and data set.

"""

bases =array(
    [
    kron(bH,bH),kron(bH,bV),kron(bV,bV),kron(bV,bH),
    kron(bR,bH),kron(bR,bV),kron(bD,bV),kron(bD,bH),
    kron(bD,bR),kron(bD,bD),kron(bR,bD),kron(bH,bD),
    kron(bV,bD),kron(bV,bL),kron(bH,bL),kron(bR,bL)
    ]
    )



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

def doIterativeAlgorithm(maxNumberOfIteration, listOfExperimentalDatas):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) to a set of datas given from a experiment.
        This recieve two variables (maxNumberOfIteration, listAsExperimentalDatas),
        and return most likely estimated density matrix (np.array).

        First quantum state matrix for this algorithm is a identity matrix.


        --------------------------------------------------------------------------------------------------------------
        Return:

            most likely estimated density matrix(np.array), time difference(datetime.timedelta)

    """

    iter = 0
    dimH = 4
    # TODO: why is epsilon so big number?
    # bigEpsilon = 10000000
    # smallEpsilon = 0.01
    epsilon = 10000000
    # epsilon = 0.01
    TolFun = 10e-11
    endDiff = 10e-10
    diff = 100
    traceDistance = 100

    dataList = listOfExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList / totalCountOfData # nDataList is a list of normarized datas

    densityMatrix = identity(dimH) # Input Density Matrix in Diluted MLE  (Identity)

    startTime = datetime.now() #Timestamp

    while traceDistance > TolFun and iter <= maxNumberOfIteration:
    # while diff > endDiff and iter <= maxNumberOfIteration:

        probList = [trace(bases[i] @ densityMatrix) for i in range(16)]
        nProbList = probList / sum(probList)
        rotationMatrix = sum([(nDataList[i] / probList[i])*bases[i] for i in range(16)])


        """ Normalization of Matrices for Measurement Bases """
        U = np.linalg.inv(sum(bases)) / sum(probList)   
        rotationMatrixLeft = (identity(dimH) + epsilon * U @ rotationMatrix) / (1 + epsilon)
        rotationMatrixRight = (identity(dimH) + epsilon * rotationMatrix @ U) / (1 + epsilon)


        """ Calculation of updated density matrix """
        modifiedDensityMatrix = rotationMatrixLeft @ densityMatrix @ rotationMatrixRight / trace(rotationMatrixLeft @ densityMatrix @ rotationMatrixRight)
        eigValueArray, eigVectors = np.linalg.eig(densityMatrix - modifiedDensityMatrix)
        traceDistance = sum(np.absolute(eigValueArray)) / 2


        """ Update Likelihood Function, and Compared with older one """
        LikelihoodFunction = sum([nDataList[i]*np.log(nProbList[i]) for i in range(16)])
        probList = [trace(bases[i] @ modifiedDensityMatrix) for i in range(16)]
        nProbList = probList / sum(probList)
        modifiedLikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(16)])
        
        diff = modifiedLikelihoodFunction - LikelihoodFunction


        """ Show Progress of Calculation """
        progress = 100 * iter / maxNumberOfIteration
        if progress % 5 == 0:
            msg = "Progress of calculation: " + str(int(progress)) + "%"
            print(msg)


        iter += 1


        """ Check Increasing of Likelihood Function  """
        if diff < 0:
            # print("--------------------------------------------------------------------")
            # print("Likelihood Function decreased. Please change the number of epsilon.")
            # print("--------------------------------------------------------------------")
            # break

            epsilon = epsilon * 0.9
            continue
        

        """ Update Density Matrix """
        densityMatrix = modifiedDensityMatrix.copy()


    endTime = datetime.now() #Timestamp
    

    """ Check That Max Iteration Number was proper """
    if iter >= maxNumberOfIteration:
        print("----------------------------------------------")
        print("Iteration time reached max iteration number.")
        print("The number of iteration times is too small.")
        print("----------------------------------------------")

    
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
    fidelity = np.real(trace(idealDensityMatrix @ estimatedDensityMatrix))

    return fidelity



if __name__ == "__main__":
    listOfExperimentalDatas = array(list(map(float, input().split())))

    maxNumberOfIteration = 10000000

    estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(maxNumberOfIteration, listOfExperimentalDatas)

    idealDensityMatrix = array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    fidelity = calculateFidelity(idealDensityMatrix, estimatedDensityMatrix)

    print(estimatedDensityMatrix)

    print("Fidelity is " + str(fidelity))

    print("Time of calculation: ",timeDifference)


