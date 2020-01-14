'''
This program is for quantum state tomography
One method is to reconstruct states with a slight correction based on optimization
following to the determination of the expectation value of a set of complete bases
It is according to 'measurement of qubits (10.1103/PhysRevA.64.052312)' for 2-qubits states.
The other method is to optimize the likelihood function based on the expectation
value of some bases (not necessary complete sets).It is according to 'Iterative
algorithm for reconstruction of entangled states (10.1103/PhysRevA.63.040303)'
for n-qubits states.
'''

data = 'width0.03_300s_1ps_5mW-w1580.481540.98-l6_count_sum.txt' #experimental data inputs
#filename_former =
#filename_latter =+

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from numpy import array, kron, genfromtxt
from scipy.optimize import minimize, minimize_scalar
import pickle

'''
method 1
'''

""" setting """
# M matrix from 'measurement of qubits'
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

# bases
bH = array([[1,0],[0,0]])
bV = array([[0,0],[0,1]])
bD = array([[1/2,1/2],[1/2,1/2]])
bR = array([[1/2,1j/2],[-1j/2,1/2]])
bL = array([[1/2,-1j/2],[1j/2,1/2]])


twoQubitsBases = array(
    [
        kron(bH,bH),kron(bH,bV),kron(bV,bV),kron(bV,bH),
        kron(bR,bH),kron(bR,bV),kron(bD,bV),kron(bD,bH),
        kron(bD,bR),kron(bD,bD),kron(bR,bD),kron(bH,bD),
        kron(bV,bD),kron(bV,bL),kron(bH,bL),kron(bR,bL)
    ]
)


# def sum_bin(filename, start, stop, delimiter=','):
#     """ sum up the bins in the selected area in a histgram """
#     n = genfromtxt(filename, delimiter)
#     return np.sum(n, axis=1)

# def sum_loop(filename_former, stop, filename_latter='.txt', start='0', step='1', delimiter=' '):
#     n = array([genfromtxt('filename_former{}filename_latter'.format(i)) for i in range(start, stop, step)])
#     return np.sum(n, axis=0)


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

class Val:
    """ calculate some function of obtained density matrix"""
    def __init__(self,ρp):
        self.ρ = ρp

    def F(self):
        """ calcuration of maximum fidelity with maximally entangled state like HH+e^iθVV """
        def EPR(θ):
            sEPR = array([1/np.sqrt(2),0,0,np.exp(1j*θ)/np.sqrt(2)])
                # HH+exp(iθ)VV : examples of maximally entangled states
            dEPR = kron(sEPR,np.conj(sEPR)).reshape(4,4)
            dEPR
            return -np.real(np.trace(self.ρ@dEPR))
        F_res = minimize_scalar(EPR, bounds=(-np.pi,np.pi), method='bounded')
        return -EPR(F_res.x), F_res.x

    def P(self):
        """ calculation of puarity"""
        return np.real(np.trace(self.ρ@self.ρ))

    def output(self):
        """ output the dendity matrix and its calculated properties """
        self.val = 'Density Matrix = \n {} \n Fidelity = {} (Angle={}), Purity = {}'.format(self.ρ,self.F()[0],self.F()[1],self.P())
        with open(data.strip('.txt')+'_val.txt', 'w', encoding='utf8') as f: # save the results
            f.write(self.val)

class Plot:
    """ plot real part and imaginary part of a density matrix"""
    def __init__(self,ρp):
        self.ρ = ρp

        """ plot setting """
        int = np.arange(4)
        _x, _y = np.meshgrid(int,int)
        x, y = _x.ravel(), _y.ravel()
        dx = dy = 0.7
        dz = self.ρ.ravel()
        z = np.zeros_like(x)
        label = array(['HH','HV','VH','VV'])

        def setting(ax):
            ax.set_xticklabels(label)
            ax.set_yticklabels(label)
            ax.set_zlim([-0.2,0.8])

        self.fig = plt.figure(figsize=(10,5))

        ax1 = self.fig.add_subplot(121, projection='3d')
        ax1.bar3d(x, y, z, dx, dy, np.real(dz), edgecolor='black')
        ax1.set_title('Real Part')
        setting(ax1)

        ax2 = self.fig.add_subplot(122, projection='3d')
        ax2.bar3d(x, y, z, dx, dy, np.imag(dz), edgecolor='black')
        ax2.set_title('Imaginary Part')
        setting(ax2)

    def pkl(self):
        """ reserve the settings and data of plot using pickle """
        with open(data.strip('.txt')+'_plot.pkl', 'wb') as f:
            pickle.dump(self.fig, f)

if __name__ == '__main__':

    n = genfromtxt(data, delimiter='\t') #experimental data inputs
#    n = sum_bin(data, start='5800', stop='6900', delimiter=',')
#    n = sum_loop(filename_former, filename_latter='.txt', stop='14')
    print(n)
    ρ = np.sum(n*M_, 2)/np.sum(n*np.trace(M_))
        #reconstructed density matrix (not physical beacause it may be nagative )

    ini = Ini(ρ)
    
    res = minimize(L, ini.t(), method='Nelder-Mead', options={'maxiter':100000, 'xatol':10**-5}) #res.x is the solution of the optimization problem
#    res = minimize(L, [1/4,1/4,1/4,1/4,0,0,0,0,0,0,0,0,0,0,0,0], method='Nelder-Mead', options={'maxiter':50000})
#    print(res.success,res.nit,L(res.x))

    """
    test_matrix = array(
    [[0.5069,-0.0239+0.0106j,-0.0412-0.0221j,0.4833+0.0329j],
    [-0.0239-0.0106j,0.0048,0.0023+0.0019j,-0.0296-0.0077j],
    [-0.0412+0.0221j,0.0023-0.0019j,0.0045,-0.0425+0.0192j],
    [0.4833-0.0329j,-0.0296+0.0077j,-0.0425-0.0192j,0.4839]])
    ini_test = Ini(test_matrix)
    print(L(ini_test.t()),L(res.x))
    """

    ρp = ρt(res.x)
    val = Val(ρp)
#    plot_raw = Plot(ρ)
#    plot_pst = Plot(ini.ρ())
    plot_opt = Plot(ρp)

    print('Fidelity = {} (Angle={}), Purity = {}'.format(val.F()[0],val.F()[1],val.P()))
    val.output()
    plt.show()
