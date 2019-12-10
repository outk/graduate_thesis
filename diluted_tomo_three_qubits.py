'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
for quantum state tomography.

This is for three qubits system.

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
matrix of bases for two qubits: 
    
    bases (numpy.array ((8*8)*64)) 


------------------
A order of this bases array is one of most important things in calculation.
So you must match each other between this and data set.

"""

bases = array(
    [
        kron(kron(bH,bH),bH),kron(kron(bH,bH),bV),kron(kron(bH,bH),bR),kron(kron(bH,bH),bD),  
        kron(kron(bH,bV),bD),kron(kron(bH,bV),bR),kron(kron(bH,bV),bV),kron(kron(bH,bV),bH),
        kron(kron(bV,bV),bH),kron(kron(bV,bV),bV),kron(kron(bV,bV),bR),kron(kron(bV,bV),bD),
        kron(kron(bV,bH),bD),kron(kron(bV,bH),bR),kron(kron(bV,bH),bV),kron(kron(bV,bH),bH),
        kron(kron(bR,bH),bH),kron(kron(bR,bH),bV),kron(kron(bR,bH),bR),kron(kron(bR,bH),bD),
        kron(kron(bR,bV),bD),kron(kron(bR,bV),bR),kron(kron(bR,bV),bV),kron(kron(bR,bV),bH),
        kron(kron(bD,bV),bH),kron(kron(bD,bV),bV),kron(kron(bD,bV),bR),kron(kron(bD,bV),bD),
        kron(kron(bD,bH),bD),kron(kron(bD,bH),bR),kron(kron(bD,bH),bV),kron(kron(bD,bH),bH),
        kron(kron(bD,bR),bH),kron(kron(bD,bR),bV),kron(kron(bD,bR),bR),kron(kron(bD,bR),bD),
        kron(kron(bD,bD),bD),kron(kron(bD,bD),bR),kron(kron(bD,bD),bV),kron(kron(bD,bD),bH),
        kron(kron(bR,bD),bH),kron(kron(bR,bD),bV),kron(kron(bR,bD),bR),kron(kron(bR,bD),bD),
        kron(kron(bH,bD),bD),kron(kron(bH,bD),bR),kron(kron(bH,bD),bV),kron(kron(bH,bD),bH),
        kron(kron(bV,bD),bH),kron(kron(bV,bD),bV),kron(kron(bV,bD),bR),kron(kron(bV,bD),bD),
        kron(kron(bV,bL),bD),kron(kron(bV,bL),bR),kron(kron(bV,bL),bV),kron(kron(bV,bL),bH),
        kron(kron(bH,bL),bH),kron(kron(bH,bL),bV),kron(kron(bH,bL),bR),kron(kron(bH,bL),bD),
        kron(kron(bR,bL),bD),kron(kron(bR,bL),bR),kron(kron(bR,bL),bV),kron(kron(bR,bL),bH)
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
    dimH = 8
    # TODO: why is epsilon so big number?
    # bigEpsilon = 10000000
    # smallEpsilon = 0.01
    epsilon = 1000
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

        probList = [trace(bases[i] @ densityMatrix) for i in range(64)]
        nProbList = probList / sum(probList)
        rotationMatrix = sum([(nDataList[i] / probList[i])*bases[i] for i in range(64)])


        """ Normalization of Matrices for Measurement Bases """
        U = np.linalg.inv(sum(bases)) / sum(probList)   
        rotationMatrixLeft = (identity(dimH) + epsilon * U @ rotationMatrix) / (1 + epsilon)
        rotationMatrixRight = (identity(dimH) + epsilon * rotationMatrix @ U) / (1 + epsilon)


        """ Calculation of updated density matrix """
        modifiedDensityMatrix = rotationMatrixLeft @ densityMatrix @ rotationMatrixRight / trace(rotationMatrixLeft @ densityMatrix @ rotationMatrixRight)
        eigValueArray, eigVectors = np.linalg.eig(densityMatrix - modifiedDensityMatrix)
        traceDistance = sum(np.absolute(eigValueArray)) / 2


        """ Update Likelihood Function, and Compared with older one """
        LikelihoodFunction = sum([nDataList[i]*np.log(nProbList[i]) for i in range(64)])
        probList = [trace(bases[i] @ modifiedDensityMatrix) for i in range(64)]
        nProbList = probList / sum(probList)
        modifiedLikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(64)])
        
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

            epsilon = epsilon * 0.1
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
    # listOfExperimentalDatas = array(list(map(float, input().split())))

    # listOfExperimentalDatas = array([500.0, 0.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0, 250.0, 0.0, 125.0, 125.0, 125.0, 125.0, 250.0, 0.0, 0.0, 250.0, 125.0, 125.0, 125.0, 125.0, 0.0, 250.0, 125.0, 125.0, 0.0, 125.0, 250.0, 125.0, 125.0, 125.0, 125.0, 125.0, 0.0, 125.0, 125.0, 125.0, 0.0, 250.0, 0.0, 250.0, 125.0, 125.0, 125.0, 125.0, 250.0, 0.0, 250.0, 0.0, 125.0, 125.0, 250.0, 125.0, 125.0, 125.0])

    listOfExperimentalDatas = array([0.0, 333.3333333333334, 166.6666666666667, 166.6666666666667, 0.0, 0.0, 0.0, 0.0, 333.3333333333334, 0.0, 166.6666666666667, 166.6666666666667, 166.6666666666667, 166.6666666666667, 0.0, 333.3333333333334, 166.6666666666667, 166.6666666666667, 333.3333333333334, 166.6666666666667, 83.33333333333336, 83.33333333333336, 0.0, 166.6666666666667, 166.6666666666667, 0.0, 83.33333333333336, 83.33333333333336, 333.3333333333334, 166.6666666666667, 166.6666666666667, 166.6666666666667, 166.6666666666667, 83.33333333333336, 208.33333333333337, 208.3333333333334, 375.0000000000001, 208.33333333333337, 83.33333333333336, 333.3333333333334, 333.3333333333334, 83.33333333333336, 375.0000000000001, 208.33333333333337, 83.33333333333336, 83.33333333333336, 166.6666666666667, 0.0, 666.6666666666669, 0.0, 333.3333333333334, 333.3333333333334, 166.6666666666667, 166.6666666666667, 0.0, 333.3333333333334, 0.0, 166.6666666666667, 83.33333333333336, 83.33333333333336, 208.33333333333337, 208.3333333333334, 83.33333333333336, 166.6666666666667])

    maxNumberOfIteration = 10000000

    estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(maxNumberOfIteration, listOfExperimentalDatas)

    # ghz = np.zeros([8,8])
    # ghz[0][0] = 1 / 2
    # ghz[7][7] = 1 / 2
    # ghz[0][7] = 1 / 2
    # ghz[7][0] = 1 / 2

    w = np.zeros([1,8])
    w[0][4] = 1 / sqrt(3)
    w[0][6] = 1 / sqrt(3)
    w[0][1] = 1 / sqrt(3)
    wmatrix = w.T @ w

    fidelity = calculateFidelity(wmatrix, estimatedDensityMatrix)

    print(estimatedDensityMatrix)

    # print(w)

    print("Fidelity is " + str(fidelity))

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


