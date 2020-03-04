import numpy as np
from numba import njit


@njit
def roll2dAxis0(matrix, n):
    result = np.empty(matrix.shape)
    for col in range(matrix.shape[1]):
        result[:,col] = np.roll(matrix[:,col], n)
    return result.astype(matrix.dtype)


@njit
def roll2dAxis1(matrix, n):
    result = np.empty(matrix.shape)
    for row in range(matrix.shape[0]):
        result[row] = np.roll(matrix[row], n)
    return result.astype(matrix.dtype)

@njit
def roll3dAxis1(matrix, n):
    result = np.empty(matrix.shape)
    for mat in range(matrix.shape[0]):
        result[mat] = roll2dAxis0(matrix[mat], n)
    return result.astype(matrix.dtype)

@njit
def roll3dAxis2(matrix, n):
    result = np.empty(matrix.shape)
    for mat in range(matrix.shape[0]):
        result[mat] = roll2dAxis1(matrix[mat], n)
    return result.astype(matrix.dtype)

