'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
for quantum state tomography.

This is for two qubits system.

'''

import numpy as np
import scipy as sp
from numpy import array, kron, genfromtxt, trace, dot, identity
from scipy.optimize import minimize, minimize_scalar


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
bases = array(
    [
    kron(bH,bH),kron(bH,bV),kron(bV,bH),kron(bV,bV),
    kron(bH,bD),kron(bH,bL),kron(bD,bH),kron(bR,bH),
    kron(bD,bD),kron(bR,bD),kron(bR,bL),kron(bD,bR),
    kron(bD,bV),kron(bR,bV),kron(bV,bD),kron(bV,bL)
    ]
    )



""" 
definition of Phase flip matrix: 
    
    Sflip (numpy.array (4*4)) 

"""
Sfilp = array(
    [
    [0, 0, 0, -1],
    [0, 0 , 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0]
    ]
    )



"""
get exparimental datas

"""

def getDatasFromFile(fileOfExparimentalDatas, numberOfQubits):
    """
    getDatasFromFile(fileOfExparimentalDatas, numberOfQubits):

        This function is getting datas from file of exparient consequense,
        and return matrix (np.array (numberOfQubits*numberOfQubits)) of them.

    """
    matrixOfExparimentalDatas = np.zeros([numberOfQubits,numberOfQubits], dtype=np.complex)
    # TODO: modify matrixOfExparimentalDatas by given data file.
    return matrixOfExparimentalDatas



"""
Iterative algorithm 

"""
def doIterativeAlgorithm(matrix, maxNumberOfIteration):
    """
    doIterativeAlgorithm():

        This function is to do iterative algorithm(10.1103/PhysRevA.63.040303) to given matrix,
        and return modified matrix (np.array) as outcome.

    """

    i = 0

    prob = [trace(dot(matrix,base[i])) for i in range(16)]
    nprob = prob/sum(prob)
    Rot = sum([ndata[i]/prob[i]*base[i] for i in range(16)])

    modifiedMatrix = matrix.copy()

    if modifiedMatrix is matrix:
        print("Matrix failed to be copyed deeply for doing iterative algorithm.")
        break

    while i <= maxNumberOfIteration:
        # TODO: modify
        """ Normalization Matrix for Measurement Bases """
        U = Inverse[sum(base)]/sum(prob)
        dRotL = (identity(dimH) + epsilon*dot(U,Rot)/(1 + epsilon)
        dRotR = (identity(dimH) + epsilon*Rot.U)/(1 + epsilon)


        """ Calculation of output density matrix """
        modifiedMatrix = dot(dot(dRotL,matrix),dRotR)/trace(dot(dot(dRotL,matrix),dRotR)
        Trdis = sum(abs(Eigenvalues[matrix - modifiedMatrix]))/2


        """ Likelihood Function """
        Lhin = Total[Table[ndata[[i]]*Log[nprob[[i]]], {i, 16}]]
        prob = Table[Tr[modifiedMatrix.base[[i]]], {i, 1, 16}]
        nprob = prob/Total[prob]
        Lhout = Total[Table[ndata[[i]]*Log[nprob[[i]]], {i, 16}]]
        diff = Re[Lhout - Lhin]

        matrix = modifiedMatrix  
        
        i += 1

    modifiedMatrix = matrixAfter

    return modifiedMatrix























if __name__ == "__main__":
    print(bases)