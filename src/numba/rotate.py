import numpy as np
from numba import njit

@njit
def rot902d(matrix):
    result = np.empty(matrix.shape)
    for row in range(matrix.shape[0]):
        result[:,row] = matrix[row][::-1]
    return result.astype(matrix.dtype)

@njit
def rot903d(matrix):
    result = np.empty(matrix.shape)
    for mat in range(matrix.shape[0]):
        result[mat] = rot902d(matrix[mat])
    return result.astype(matrix.dtype)
