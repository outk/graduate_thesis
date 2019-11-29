'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
for quantum state tomography.

This is for two qubits system.

'''

import numpy as np
from numpy import array, kron, trace, dot, identity


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

"""
# bases = array(
#     [
#     kron(bH,bH),kron(bH,bV),kron(bV,bH),kron(bV,bV),
#     kron(bH,bD),kron(bH,bL),kron(bD,bH),kron(bR,bH),
#     kron(bD,bD),kron(bR,bD),kron(bR,bL),kron(bD,bR),
#     kron(bD,bV),kron(bR,bV),kron(bV,bD),kron(bV,bL)
#     ]
#     )

bases =array(
    [
    kron(bH,bH),kron(bH,bV),kron(bV,bV),kron(bV,bH),
    kron(bR,bH),kron(bR,bV),kron(bD,bV),kron(bD,bH),
    kron(bD,bR),kron(bD,bD),kron(bR,bD),kron(bH,bD),
    kron(bV,bD),kron(bV,bL),kron(bH,bL),kron(bR,bL)
    ]
    )


"""
get experimental datas

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
Iterative algorithm 

"""
def doIterativeAlgorithm(matrix, maxNumberOfIteration, listAsExperimentalDatas):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) to given matrix,
        and return modified matrix (np.array) as outcome.

    """

    i = 0
    dimH = 4
    m = 0
    n = 1
    s = 1
    epsilon = 10000000
    TolFun = 10e-10
    dimH = 4
    m = 0

    dataList = listAsExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList/totalCountOfData # nDataList is a list of normarized data

    # matrix = np.identity(4) # Input Density Matrix in Diluted MLE  (Identity)
    diff = 100
    Trdis = 100

    modifiedMatrix = matrix.copy()
    if modifiedMatrix is matrix:
        print("Matrix failed to be copyed deeply for doing iterative algorithm.")
        return -1

    while Trdis > TolFun and i <= maxNumberOfIteration:

        probList = [trace(dot(matrix,bases[i])) for i in range(16)]
        nProbList = probList/sum(probList)
        # TODO: Why is probList used not nProbList here?
        Rot = sum([(nDataList[i]/probList[i])*bases[i] for i in range(16)])


        """ Normalization Matrix for Measurement Bases """
        U = np.linalg.inv(sum(bases))/sum(probList)
        dRotL = (identity(dimH) + epsilon*dot(U,Rot))/(1 + epsilon)
        dRotR = (identity(dimH) + epsilon*dot(Rot,U))/(1 + epsilon)


        """ Calculation of output density matrix """
        modifiedMatrix = dot(dot(dRotL,matrix),dRotR)/trace(dot(dot(dRotL,matrix),dRotR))
        eigValueArray, eigVectors = np.linalg.eig(matrix - modifiedMatrix)
        Trdis = sum(np.absolute(eigValueArray))/2


        """ Likelihood Function """
        Lhin = sum([nDataList[i]*np.log(nProbList[i]) for i in range(16)])
        probList = [trace(dot(modifiedMatrix,bases[i])) for i in range(16)]
        nProbList = probList/sum(probList)
        Lhout = sum([nDataList[i]*(np.log(nProbList[i])) for i in range(16)])
        diff = np.real(Lhout - Lhin)

        matrix = modifiedMatrix.copy()
        if modifiedMatrix is matrix:
            print("Matrix failed to be copyed deeply for doing iterative algorithm.")
            return -1
        
        i += 1

    return modifiedMatrix





if __name__ == "__main__":
    listOfExperimentalDatas = array([101599, 17900, 92913, 3263, 52733, 56453, 50266, 57325, 52234, 18951, 49106, 59811, 54621, 47110, 62711, 18847])

    matrix = np.identity(4)/4

    maxNumberOfIteration = 10000

    finalMatrix = doIterativeAlgorithm(matrix, maxNumberOfIteration, listOfExperimentalDatas)

    print(finalMatrix)