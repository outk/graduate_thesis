'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
for quantum state tomography.

This is for qutrits system.

'''

import numpy as np
from numpy import array, kron, trace, identity, sqrt, random, zeros, exp, pi, conjugate, random
import scipy
from scipy.linalg import sqrtm, funm
from datetime import datetime, timedelta
from concurrent import futures


""" 
definition of three frequency bases: 

    fb1 = array([1, 0, 0])
    fb2 = array([0, 1, 0])
    fb3 = array([0, 0, 1])

"""

zero_base_array1 = zeros((1,3))
zero_base_array1[0][0] = 1
fb1 = zero_base_array1

zero_base_array2 = zeros((1,3))
zero_base_array2[0][1] = 1
fb2 = zero_base_array2

zero_base_array3 = zeros((1,3))
zero_base_array3[0][2] = 1
fb3 = zero_base_array3



"""
make measurement bases

"""

mb1 = (conjugate((fb1 + fb2).T) @ (fb1 + fb2)) / 2
mb2 = (conjugate((fb1 + fb3).T) @ (fb1 + fb3)) / 2
mb3 = (conjugate((fb2 + fb3).T) @ (fb2 + fb3)) / 2
mb4 = (conjugate((exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3)) * fb2).T) @ (exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3) * fb2))) / 2
mb5 = (conjugate((exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3)) * fb2).T) @ (exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3) * fb2))) / 2
mb6 = (conjugate((exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3)) * fb3).T) @ (exp( 2*pi*1j/3) * fb1 + (exp(-2*pi*1j/3) * fb3))) / 2
mb7 = (conjugate((exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3)) * fb3).T) @ (exp(-2*pi*1j/3) * fb1 + (exp( 2*pi*1j/3) * fb3))) / 2
mb8 = (conjugate((exp( 2*pi*1j/3) * fb2 + (exp(-2*pi*1j/3)) * fb3).T) @ (exp( 2*pi*1j/3) * fb2 + (exp(-2*pi*1j/3) * fb3))) / 2
mb9 = (conjugate((exp(-2*pi*1j/3) * fb2 + (exp( 2*pi*1j/3)) * fb3).T) @ (exp(-2*pi*1j/3) * fb2 + (exp( 2*pi*1j/3) * fb3))) / 2

bases = array([mb1, mb2, mb3, mb4, mb5, mb6, mb7, mb8, mb9])


def makeBases(numberOfQutrits, bases):
    for _ in range(numberOfQutrits-1):
        fixedBases = []
        for base1 in bases:
            fixedBases.extend([kron(base1, base2) for base2 in bases])
        bases = fixedBases.copy()

    return array(bases)



"""
Get Experimental Datas

"""

def getDatasFromFile(fileOfExperimentalDatas, numberOfQutrits):
    """
    getDatasFromFile(fileOfExperimentalDatas, numberOfQutrits):

        This function is getting datas from file of experiment consequense,
        and return matrix (np.array (numberOfQutrits*numberOfQutrits)) of them.

    """
    matrixOfExperimentalDatas = np.zeros([numberOfQutrits,numberOfQutrits], dtype=np.complex)
    # TODO: modify matrixOfExperimentalDatas by given data file.
    return matrixOfExperimentalDatas



"""
Iterative Algorithm 

"""

def doIterativeAlgorithm(numberOfQutrits, bases, maxNumberOfIteration, listOfExperimentalDatas):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) and diluted MLE algorithm(10.1103/PhysRevA.75.042108) to a set of datas given from a experiment.
        This recieve four variables (numberOfQutrits, bases, maxNumberOfIteration, listAsExperimentalDatas),
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

    densityMatrix = identity(3 ** numberOfQutrits) # Input Density Matrix in Diluted MLE  (Identity)

    startTime = datetime.now() #Timestamp

    
    """ Start iteration """
    while traceDistance > TolFun and iter <= maxNumberOfIteration:
    # while diff > endDiff and iter <= maxNumberOfIteration:

        probList = [trace(bases[i] @ densityMatrix) for i in range(len(bases))]
        nProbList = probList / sum(probList)
        rotationMatrix = sum([(nDataList[i] / probList[i]) * bases[i] for i in range(9 ** numberOfQutrits)])


        """ Normalization of Matrices for Measurement Bases """
        U = np.linalg.inv(sum(bases)) / sum(probList)   
        rotationMatrixLeft = (identity(3 ** numberOfQutrits) + epsilon * U @ rotationMatrix) / (1 + epsilon)
        rotationMatrixRight = (identity(3 ** numberOfQutrits) + epsilon * rotationMatrix @ U) / (1 + epsilon)


        """ Calculation of updated density matrix """
        modifiedDensityMatrix = rotationMatrixLeft @ densityMatrix @ rotationMatrixRight / trace(rotationMatrixLeft @ densityMatrix @ rotationMatrixRight)
        eigValueArray, eigVectors = np.linalg.eig(densityMatrix - modifiedDensityMatrix)
        traceDistance = sum(np.absolute(eigValueArray)) / 2


        """ Update Likelihood Function, and Compared with older one """
        LikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(9 ** numberOfQutrits)])
        probList = [trace(bases[i] @ modifiedDensityMatrix) for i in range(len(bases))]
        nProbList = probList / sum(probList)
        modifiedLikelihoodFunction = sum([nDataList[i] * np.log(nProbList[i]) for i in range(9 ** numberOfQutrits)])
        
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



""" Poisson Distributed Simulation """

def doPoissonDistirbutedSimulation(numberOfQutrits, bases, maxNumberOfIteration, experimentalData, idealMatrix):
    """
    doPoissonDistributedSimulation(experimentalData, iterationTime)

    """

    estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(numberOfQutrits, bases, maxNumberOfIteration, random.poisson(experimentalData))
    fidelity = calculateFidelity(idealMatrix, estimatedDensityMatrix)

    with open('test_qutrit_tomo_fidelity.txt', mode='a') as f:
        f.write(str(fidelity) + '\n')



if __name__ == "__main__":
    start_time = datetime.now()

    with open('test.txt') as f:
        listOfExperimentalData = array([2+int(s.strip()) for s in f.readlines()])

    maxNumberOfIteration = 10000000

    numberOfQutrits = 2

    newbases = makeBases(numberOfQutrits, bases)

    # estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(numberOfQutrits, newbases, maxNumberOfIteration, listOfExperimentalDatas)


    """ example of two qutrits """
    baseVecter = np.zeros([1, 3**numberOfQutrits])
    baseVecter[0][0] = 1 / sqrt(2)
    baseVecter[0][3**numberOfQutrits-1] = 1 / sqrt(2)
    matrix = baseVecter.T @ baseVecter
    

    """ Pallarel Computing """

    with futures.ProcessPoolExecutor(max_workers=3) as executor:
        for _ in range(100):
            executor.submit(fn=doPoissonDistirbutedSimulation, numberOfQutrits=numberOfQutrits, bases=newbases, maxNumberOfIteration=maxNumberOfIteration, experimentalData=listOfExperimentalData, idealMatrix=matrix)


    end_time = datetime.now()

    print(end_time - start_time)
    


