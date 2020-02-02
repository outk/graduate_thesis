import numpy as np

numberOfQubits = 4

matrix = np.zeros([2**numberOfQubits, 2**numberOfQubits])
baseVecter = np.zeros([1, 2**numberOfQubits])
baseVecter[0][1] = 1
matrix += baseVecter.T @ baseVecter
baseVecter = np.zeros([1, 2**numberOfQubits])
baseVecter[0][2] = 1
matrix += baseVecter.T @ baseVecter
baseVecter = np.zeros([1, 2**numberOfQubits])
baseVecter[0][4] = 1
matrix += baseVecter.T @ baseVecter
baseVecter = np.zeros([1, 2**numberOfQubits])
baseVecter[0][8] = 1
matrix += baseVecter.T @ baseVecter

baseVecter = np.zeros([1, 2**numberOfQubits])
baseVecter[0][0] = 1
baseVecter[0][2**numberOfQubits-1] = 1
matrix += baseVecter.T @ baseVecter

matrix = matrix/np.trace(matrix)

print(matrix)