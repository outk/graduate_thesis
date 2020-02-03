import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from numpy import array, kron, genfromtxt
from scipy.optimize import minimize, minimize_scalar
import pickle


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

if __name__ == "__main__":
    