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


su2b = array([
    [[   1,  0], [   0,  1]],
    [[   0,  1], [   1,  0]],
    [[   0,-1j], [  1j,  0]],
    [[   1,  0], [   0, -1]]
]
)

def makeSU2Bases(numberOfQubits):
    newbases = su2b.copy()
    su2Bases = []
    for i in range(numberOfQubits-1):
        for newbase in newbases:
            su2Bases.extend([kron(newbase, base) for base in su2b])
        newbases = su2Bases.copy()
        su2Bases = []

    return array(su2Bases)


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


def makeBMatrix(numberOfQubits, bases, su2Bases):
    B = np.zeros((4**numberOfQubits, 4**numberOfQubits))

    for i in range(4**numberOfQubits):
        for j in range(4**numberOfQubits):
            B[i][j] = np.trace(bases[i] @ su2Bases[j])

    return B

def makeRList(numberOfQubits, dataList, bases, su2Bases):
    B = makeBMatrix(numberOfQubits, bases, su2Bases)

    BInverse = np.linalg.inv(B)

    rlist = np.array([sum([BInverse[i][j] * dataList[j] for j in range(4**numberOfQubits)]) for i in range(4**numberOfQubits)])

    return rlist

def makeInitialDensityMatrix(numberOfQubits, dataList, bases, su2Bases):
    rList = makeRList(numberOfQubits, dataList, bases, su2Bases)

    densityMatrix = sum([rList[i] * su2Bases[i] for i in range(4**numberOfQubits)])
    densityMatrix = densityMatrix / np.trace(densityMatrix)

    return densityMatrix


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



def doIterativeAlgorithm(numberOfQubits, bases, listOfExperimentalDatas, su2Bases):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) and diluted MLE algorithm(10.1103/PhysRevA.75.042108) to a set of datas given from a experiment.
        This recieve four variables (numberOfQubits, bases, maxNumberOfIteration, listAsExperimentalDatas),
        and return most likely estimated density matrix (np.array) and total time of calculation(datetime.timedelta).

        First quantum state matrix for this algorithm is a identity matrix.


        --------------------------------------------------------------------------------------------------------------
        Return:

            most likely estimated density matrix(np.array), 

    """

    """ Setting initial parameters """
    iter = 0
    epsilon = 1000
    TolFun = 10e-11
    # endDiff = 10e-10
    # diff = 100
    traceDistance = 100
    maxNumberOfIteration = 100000

    dataList = listOfExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList / totalCountOfData # nDataList is a list of normarized datas
    densityMatrix = makeInitialDensityMatrix(numberOfQubits, dataList, bases, su2Bases)

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

    return modifiedDensityMatrix



""" Calculate Fidelity """

def calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):
    """
    calculateFidelity(idealDensityMatrix, estimatedDensityMatrix):

    """

    fidelity = np.real(trace(sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix)) @ sqrtm(sqrtm(idealDensityMatrix) @ estimatedDensityMatrix @ sqrtm(idealDensityMatrix))))

    return fidelity



""" Iterative Simulation """

def doIterativeSimulation(numberOfQubits, bases, pathOfExperimentalData, idealDensityMatrix, resultDirectoryName, su2Bases):
    """
    doIterativeSimulation(numberOfQubits, bases, pathOfExperimentalData, idealDensityMatrix, resultDirectoryName, su2Bases)


    """

    """ Get Experimental Data"""
    listOfExperimentalData = getExperimentalData(pathOfExperimentalData)

    """ Calculate """
    estimatedDensityMatrix = doIterativeAlgorithm(numberOfQubits, bases, listOfExperimentalData, su2Bases)
    fidelity = calculateFidelity(idealDensityMatrix, estimatedDensityMatrix)

    """ Make File Name of result """
    l = 0
    r = len(pathOfExperimentalData)-1
    for i in range(len(pathOfExperimentalData)):
        if pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == ".":
            r = len(pathOfExperimentalData)-1-i
        if pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == "/" or pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == "\\":
            l = len(pathOfExperimentalData)-i
            break
    resultFileName = pathOfExperimentalData[l:r]
    resultFilePath = '.\\result\\qubit\\iterative\\' + resultDirectoryName + '\\' + resultFileName + '_result' + '.txt'

    """ Save Result """
    with open(resultFilePath, mode='a') as f:
        f.write(str(fidelity) + '\n')



""" Poisson Distributed Simulation """

def doPoissonDistributedSimulation(numberOfQubits, bases, pathOfExperimentalData, idealDensityMatrix, resultDirectoryName, su2Bases):
    """
    doPoissonDistributedSimulation(numberOfQubits, bases, pathOfExperimentalData, idealDensityMatrix, resultDirectoryName, su2Bases)


    """

    """ Get Experimental Data"""
    listOfExperimentalData = getExperimentalData(pathOfExperimentalData)

    """ Calculate """
    estimatedDensityMatrix = doIterativeAlgorithm(numberOfQubits, bases, random.poisson(listOfExperimentalData), su2Bases)
    fidelity = calculateFidelity(idealDensityMatrix, estimatedDensityMatrix)

    """ Make File Name of result """
    l = 0
    r = len(pathOfExperimentalData)-1
    for i in range(len(pathOfExperimentalData)):
        if pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == ".":
            r = len(pathOfExperimentalData)-1-i
        if pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == "/" or pathOfExperimentalData[len(pathOfExperimentalData)-1-i] == "\\":
            l = len(pathOfExperimentalData)-i
            break
    resultFileName = pathOfExperimentalData[l:r]
    resultFilePath = '.\\result\\qubit\\poisson\\' + resultDirectoryName + "\\" + resultFileName + '_result' + '.txt'

    """ Save Result """
    with open(resultFilePath, mode='a') as f:
        f.write(str(fidelity) + '\n')



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



""" Get Paths of Experimental Data """

def getExperimentalDataPaths():
    """
    getExperimentalDataPaths()

    """

    print("------------------------------------------------------------")
    print("PLEASE ENTER PATHS OF EXPERIMENTAL DATA")
    print("")
    print("IF THERE ARE MULTIPLE DATA FILE YOU WANT TO TOMOGRAPHY,")
    print("ENTER ALL PATHS SEPARATED WITH SPACE.")
    print("LIKE THIS >> .\\datadirectory\\ex1.txt .\\datadirectory\\ex2.txt ...")
    print("------------------------------------------------------------")
    print(">>")

    paths = list(input().split())

    return paths



""" Get Name of Result Directory AND FILE """

def getNameOfResultDirectory():
    """
    getNameOfResultDirectory()

    """

    print("------------------------------------------------------------")
    print("PLEASE ENTER NAME OF RESULT DIRECTORY ")
    print("")
    print("THE RESULT DATA WILL SAVED AT ")
    print("'.\\result\\qubit\\iterative(or poisson)\\{ YOUR ENTED DIRECTORY NAME }\\{ EXPERIMENTAL DATA FILE NAME }_result.txt'")
    print("")
    print("IF EMPTY, THE NAME OF RESULT DIRECTORY IS 'default'")
    print("------------------------------------------------------------")
    print(">>")

    nameOfResultDirectory = input()
    if nameOfResultDirectory == "":
        nameOfResultDirectory = "default"

    return nameOfResultDirectory



""" Whether Do Poisson Distributed Simulation """

def checkPoisson():
    """
    checkPoisson()

    """

    print("------------------------------------------------------------")
    print("PLEASE ENTER ANSWER WHETHER DO POISSON DISTRIBUTED SIMULATION")
    print("IF YOU DO, PLEASE ENTER 'yes'")
    print("IF YOU ENTER ANOTHER WORD OR EMPTY, YOUR ANSWER IS REGARED AS 'no'")
    print("------------------------------------------------------------")
    print(">>")

    answer = input()
    if answer == "yes" or answer == "Yes" or answer == "YES":
        print("YOUR ANSWER IS: 'yes'")
        poissonPaths = getExperimentalDataPaths()
        eachIterationTime = getEachIterationTime()
        return True, poissonPaths*eachIterationTime
    else:
        print("YOUR ANSWER IS: 'no'")
        return False, []



""" Get Each Iteration Time """

def getEachIterationTime():
    """
    getEachIterationTime()

    """

    print("------------------------------------------------------------")
    print("PLEASE ENTER ITERATION TIME OF EACH POISSON SIMULATION")
    print("------------------------------------------------------------")
    print(">>")
    
    eachIterationTime = input()
    if eachIterationTime == "":
        eachIterationTime = 0
    else:
        eachIterationTime = int(eachIterationTime)

    return eachIterationTime



""" Get Number of Parallel Comuting """

def getNumberOfParallelComputing():
    """
    getNumberOfParallelComputing()

    """

    print("------------------------------------------------------------")
    print("HOW MANY TIMES DO YOU WANT TO PARALLELIZE?")
    print("IF THE NUMBER IS TOO LARGE, THE PARFORMANCE OF SIMULATION BECOME LOWER.")
    print("THE NUMBER OF LOGICAL PROCESSOR OF YOUR COMPUTER IS >>")
    print(os.cpu_count())
    print("RECOMENDED NUMBER IS LESS THAN THE ABOVE NUMBER.")
    print("------------------------------------------------------------")
    print(">>")
    
    numberOfParallelComputing = int(input())

    return numberOfParallelComputing




if __name__ == "__main__":

    """ Get Number of Qubits """
    numberOfQubits = getNumberOfQubits()

    """ Make SU2 Bases """
    su2Bases = makeSU2Bases(numberOfQubits)
    
    """ Get Paths of Experimental Data """
    paths = getExperimentalDataPaths()

    """ Get Name of Result Directory """
    resultDirectoryName = getNameOfResultDirectory()

    """ Check Poisson Distributed Simulation """
    check, poissonPaths = checkPoisson()

    """ Get Number of Parallel Computing """
    numberOfParallelComputing = getNumberOfParallelComputing()

    """ Make Bases """
    basesOfQubits = makeBases(numberOfQubits)

    """ Make Ideal Density Matrix """
    baseVecter = np.zeros([1, 2**numberOfQubits])
    baseVecter[0][0] = 1 / sqrt(2)
    baseVecter[0][2**numberOfQubits-1] = 1 / sqrt(2)
    idealDensityMatrix = baseVecter.T @ baseVecter

    start_time = datetime.now() #time stamp

    """ Make Result Directory """
    if not os.path.exists('.\\result\\qubit\\iterative\\' + resultDirectoryName):
        os.makedirs('.\\result\\qubit\\iterative\\' + resultDirectoryName)

    """ Start Tomography """
    with futures.ProcessPoolExecutor(max_workers=numberOfParallelComputing) as executor:
        for path in paths:
            executor.submit(fn=doIterativeSimulation, numberOfQubits=numberOfQubits, bases=basesOfQubits, pathOfExperimentalData=path, idealDensityMatrix=idealDensityMatrix, resultDirectoryName=resultDirectoryName, su2Bases=su2Bases)

    """ Start Poisson Distributed Simulation """
    if check:
        """ Make Result Directory for Poisson Distributed Simulation """
        if not os.path.exists('.\\result\\qubit\\poisson\\' + resultDirectoryName):
            os.makedirs('.\\result\\qubit\\poisson\\' + resultDirectoryName)

        with futures.ProcessPoolExecutor(max_workers=numberOfParallelComputing) as executor:
            for poissonPath in poissonPaths:
                executor.submit(fn=doPoissonDistributedSimulation, numberOfQubits=numberOfQubits, bases=basesOfQubits, pathOfExperimentalData=poissonPath, idealDensityMatrix=idealDensityMatrix, resultDirectoryName=resultDirectoryName, su2Bases=su2Bases)


    end_time = datetime.now() #time stamp
    print("Total Calculation Time was " + str(end_time - start_time))

    


