import numpy as np
from numpy import array, kron, trace, identity, conj, transpose
from scipy.linalg import sqrtm, funm, inv
from numpy.linalg import cholesky 
import scipy
from datetime import datetime, timedelta 
from scipy.optimize import minimize, minimize_scalar


# root2 = np.roots(2)

# b0 = array([[1,0],[0,0]])
# bp = array([[0.5,0.5],[0.5,0.5]])

# def calculate():
#     h = b0 + bp
#     hi = inv(h)
#     hir = sqrtm(hi)
#     return hir@(bp + b0)@hir , h

# if __name__ == "__main__":
#     a, h = calculate()
#     print(a)
#     print(h)


"""
--------------------------------------------------------------------------------------------------------
"""

M = array(
    [
    [[1, -(1-1j)/2, -(1+1j)/2,       1/2],  [-(1+1j)/2,     0,      1j/2,         0],   [-(1-1j)/2,     -1j/2,      0,         0],  [      1/2,         0,         0,       0]],
    [[0, -(1-1j)/2,         0,       1/2],  [-(1+1j)/2,     1,      1j/2, -(1+1j)/2],   [        0,     -1j/2,      0,         0],  [      1/2, -(1-1j)/2,         0,       0]], #42
    [[0,         0,         0,       1/2],  [        0,     0,      1j/2, -(1+1j)/2],   [        0,     -1j/2,      0, -(1-1j)/2],  [      1/2, -(1-1j)/2, -(1+1j)/2,       1]],
    [[0,         0, -(1+1j)/2,       1/2],  [        0,     0,      1j/2,         0],   [-(1-1j)/2,     -1j/2,      1, -(1-1j)/2],  [      1/2,         0, -(1+1j)/2,       0]],
    [[0,         0,        1j, -(1+1j)/2],  [        0,     0,  (1-1j)/2,         0],   [      -1j,  (1+1j)/2,      0,         0],  [-(1-1j)/2,         0,         0,       0]],
    [[0,         0,         0, -(1+1j)/2],  [        0,     0,  (1-1j)/2,        1j],   [        0,  (1+1j)/2,      0,         0],  [-(1-1j)/2,       -1j,         0,       0]],
    [[0,         0,         0, -(1+1j)/2],  [        0,     0, -(1-1j)/2,         1],   [        0, -(1+1j)/2,      0,         0],  [-(1-1j)/2,         1,         0,       0]],
    [[0,         0,         1, -(1+1j)/2],  [        0,     0, -(1-1j)/2,         0],   [        1, -(1+1j)/2,      0,         0],  [-(1-1j)/2,         0,         0,       0]],
    [[0,         0,         0,        1j],  [        0,     0,       -1j,         0],   [        0,        1j,      0,         0],  [      -1j,         0,         0,       0]], 
    [[0,         0,         0,         1],  [        0,     0,         1,         0],   [        0,         1,      0,         0],  [        1,         0,         0,       0]],
    [[0,         0,         0,        1j],  [        0,     0,        1j,         0],   [        0,       -1j,      0,         0],  [      -1j,         0,         0,       0]],
    [[0,         1,         0, -(1+1j)/2],  [        1,     0, -(1+1j)/2,         0],   [        0, -(1-1j)/2,      0,         0],  [-(1-1j)/2,         0,         0,       0]],
    [[0,         0,         0, -(1+1j)/2],  [        0,     0, -(1+1j)/2,         0],   [        0, -(1-1j)/2,      0,         1],  [-(1-1j)/2,         0,         1,       0]],
    [[0,         0,         0, -(1-1j)/2],  [        0,     0,  (1-1j)/2,         0],   [        0,  (1+1j)/2,      0,       -1j],  [-(1+1j)/2,         0,        1j,       0]], #23 32
    [[0,       -1j,         0, -(1-1j)/2],  [       1j,     0,  (1-1j)/2,         0],   [        0,  (1+1j)/2,      0,         0],  [-(1+1j)/2,         0,         0,       0]],
    [[0,         0,         0,         1],  [        0,     0,        -1,         0],   [        0,        -1,      0,         0],  [        1,         0,         0,       0]]
    ]
    )

M_ = np.moveaxis(M, 0, -1)


bH = array([[1,0],[0,0]])
bV = array([[0,0],[0,1]])
bD = array([[1/2,1/2],[1/2,1/2]])
bR = array([[1/2,1j/2],[-1j/2,1/2]])
bL = array([[1/2,-1j/2],[1j/2,1/2]])


bases = array(
    [
        kron(bH,bH),kron(bH,bV),kron(bV,bV),kron(bV,bH),
        kron(bR,bH),kron(bR,bV),kron(bD,bV),kron(bD,bH),
        kron(bD,bR),kron(bD,bD),kron(bR,bD),kron(bH,bD),
        kron(bV,bD),kron(bV,bL),kron(bH,bL),kron(bR,bL)
    ]
)


class Ini:
    """ This class generate some value from an initial density matrix """
    def __init__(self, matrix):
        self.m = matrix

    def T(self):
        """ initial T matrix for positive density matrix (measurement of qubits(4.6)) """
        T = np.zeros([4,4], dtype=np.complex)
        Δ = np.linalg.det(self.m)
        M_11 = np.linalg.det(self.m[np.ix_([1,2,3],[1,2,3])])
        M_12 = np.linalg.det(self.m[np.ix_([1,2,3],[0,2,3])])
        M_1122 = np.linalg.det(self.m[np.ix_([2,3],[2,3])])
        M_1123 = np.linalg.det(self.m[np.ix_([2,3],[1,3])])
        M_1223 = np.linalg.det(self.m[np.ix_([2,3],[0,3])])

        T[0,0] = np.real(np.sqrt(Δ/M_11))
        T[1,1] = np.real(np.sqrt(M_11/M_1122))
        T[2,2] = np.real(np.sqrt(M_1122/self.m[3,3]))
        T[3,3] = np.real(np.sqrt(self.m[3,3]))
        T[1,0] = M_12/np.sqrt(M_11*M_1122)
        T[2,1] = M_1123/np.sqrt(self.m[3,3]*M_1122)
        T[3,2] = self.m[3,2]/np.sqrt(self.m[3,3])
        T[2,0] = M_1223/np.sqrt(self.m[3,3]*M_1122)
        T[3,1] = self.m[3,1]/np.sqrt(self.m[3,3])
        T[3,0] = self.m[3,0]/np.sqrt(self.m[3,3])
        return T

    def ρ(self):
        """ positivized density matrix """
        ρ = np.conj(self.T()).T@self.T()/np.trace(np.conj(self.T()).T@self.T())
        return ρ

    def t(self):
        """ parameter t of T matrix"""
        t = [np.real(self.T()[0,0]),np.real(self.T()[1,1]),np.real(self.T()[2,2]),np.real(self.T()[3,3]),
            np.real(self.T()[1,0]),np.imag(self.T()[1,0]),np.real(self.T()[2,1]),np.imag(self.T()[2,1]),
            np.real(self.T()[3,2]),np.imag(self.T()[3,2]),np.real(self.T()[2,0]),np.imag(self.T()[2,0]),
            np.real(self.T()[3,1]),np.imag(self.T()[3,1]),np.real(self.T()[3,0]),np.imag(self.T()[3,0])]
        return t


'''
This is a implementation of 
'Iterative algorithm for reconstruction of entangled states(10.1103/PhysRevA.63.040303)'
'Diluted maximum-likelihood algorithm for quantum tomography(10.1103/PhysRevA.75.042108)'
for quantum state tomography.

This is for two qubits system.

'''


""" 
definition of base: 

    bH (numpy.array (2*2))
    bV (        "        )
    bD (        "        )
    bR (        "        )
    bL (        "        )

"""

# bH = array([[1,0],[0,0]])
# bV = array([[0,0],[0,1]])
# bD = array([[1/2,1/2],[1/2,1/2]])
# bR = array([[1/2,1j/2],[-1j/2,1/2]])
# bL = array([[1/2,-1j/2],[1j/2,1/2]])



""" 
matrix of bases for two qubits: 
    
    bases (numpy.array ((4*4)*16)) 


------------------
A order of this bases array is one of most important things in calculation.
So you must match each other between this and data set.

"""

# bases =array(
#     [
#     kron(bH,bH),kron(bH,bV),kron(bV,bV),kron(bV,bH),
#     kron(bR,bH),kron(bR,bV),kron(bD,bV),kron(bD,bH),
#     kron(bD,bR),kron(bD,bD),kron(bR,bD),kron(bH,bD),
#     kron(bV,bD),kron(bV,bL),kron(bH,bL),kron(bR,bL)
#     ]
#     )



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

def doIterativeAlgorithm(maxNumberOfIteration, listOfExperimentalDatas, initialDensityMatrix):
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
    epsilon = 1000
    # epsilon = 0.01
    TolFun = 10e-11
    endDiff = 10e-10
    diff = 100
    traceDistance = 100

    dataList = listOfExperimentalDatas
    totalCountOfData = sum(dataList)
    nDataList = dataList / totalCountOfData # nDataList is a list of normarized datas

    # densityMatrix = identity(dimH) # Input Density Matrix in Diluted MLE  (Identity)

    densityMatrix = initialDensityMatrix

    startTime = datetime.now() #Timestamp

    while traceDistance > TolFun and iter <= maxNumberOfIteration:
    # while diff > endDiff and iter <= maxNumberOfIteration:
    # while  iter <= maxNumberOfIteration:

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


def ρt(t):
    """ reconstructed density matrix from the parameters of T matrix """
    T = np.zeros([4,4], dtype=np.complex)
    T[0,0] = t[0]
    T[1,1] = t[1]
    T[2,2] = t[2]
    T[3,3] = t[3]
    T[1,0] = t[4]+t[5]*1j
    T[2,1] = t[6]+t[7]*1j
    T[3,2] = t[8]+t[9]*1j
    T[2,0] = t[10]+t[11]*1j
    T[3,1] = t[12]+t[13]*1j
    T[3,0] = t[14]+t[15]*1j
    ρt = np.conj(T).T@T/np.trace(np.conj(T).T@T)
    return ρt


def L(t):
    """ likelihood function """
    exp_Nρp = np.sum(n*np.trace(M_))*np.trace(ρt(t)@bases,axis1=1,axis2=2)
    L = np.real(np.sum((exp_Nρp-n)**2/(2*exp_Nρp)))
    return L



if __name__ == '__main__':
    n = array(list(map(int, input().split())))
    # 34749 324 35805 444 16324 17521 13441 16901 17932 32028 15132 17238 13171 17170 16722 33586

    sTime = datetime.now()
    
    ρ = np.sum(n*M_, 2)/np.sum(n*np.trace(M_))
        #reconstructed density matrix (not physical beacause it may be nagative )

    # print(ρ)

    # print("---------------------------")

    ini = Ini(ρ)

    # matrix = ini.T()

    # mMatrix = matrix.T.conj() @ matrix / np.trace(matrix.T.conj() @ matrix)

    # print(mMatrix)

    # print("------------------------------")
    
    # print(cholesky(ρ).T.conj() @ cholesky(ρ))

    # maxNumberOfIteration = 10000000

    # initialDensityMatrix = mMatrix

    # initialDensityMatrix = identity(4) / 4

    # estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(maxNumberOfIteration, n, mMatrix)

    # estimatedDensityMatrix, timeDifference = doIterativeAlgorithm(maxNumberOfIteration, n, initialDensityMatrix)

    res = minimize(L, ini.t(), method='Nelder-Mead', options={'maxiter':100000, 'xatol':10**-5})

    # print(estimatedDensityMatrix)

    endTime = datetime.now()

    allTimeDifference = endTime - sTime

    # print(res)

    print("Time of calculation: ",allTimeDifference)

    # beforeDensityMatrix = identity(4)/4

    # fidelity = calculateFidelity(beforeDensityMatrix, estimatedDensityMatrix)

    # print("Fidelity of identity is " + str(fidelity))

    # beforeDensityMatrix = mMatrix

    # fidelity = calculateFidelity(beforeDensityMatrix, estimatedDensityMatrix)

    # print("Fidelity of rhop is " + str(fidelity))
   