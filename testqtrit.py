import numpy as np
from numpy import array, sqrt, zeros, pi, exp, conjugate, kron

zero_base_array1 = zeros((1,3))
zero_base_array1[0][0] = 1
fb1 = zero_base_array1

zero_base_array2 = zeros((1,3))
zero_base_array2[0][1] = 1
fb2 = zero_base_array2

zero_base_array3 = zeros((1,3))
zero_base_array3[0][2] = 1
fb3 = zero_base_array3

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

if __name__ == "__main__":
    
    numberOfQutrits = 2

    newbases = makeBases(2, bases)

    baseVecter = np.zeros([1, 3**numberOfQutrits])
    baseVecter[0][0] = 1 / sqrt(2)
    baseVecter[0][3**numberOfQutrits-1] = 1 / sqrt(2)
    matrix = baseVecter.T @ baseVecter

    datalist = []

    for base in newbases:
        data = np.trace(base @ matrix)
        with open('test.txt', mode='a') as f:
            f.write(str(int(np.real(data)*1000)) + '\n')