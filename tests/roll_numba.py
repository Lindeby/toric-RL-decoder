import numpy as np
import time
from numba import njit, prange


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


if __name__ == "__main__":
    a = np.arange(81).reshape(9,9)
    b = np.arange(2*81).reshape(2,9,9)
    # a = np.arange(9).reshape(3,3)
    # b = np.arange(2*9).reshape(2,3,3)


    # Call once for compile
    roll2dAxis1(a, -1)
    roll3dAxis2(b, -1)
    roll2dAxis0(a, -1)
    roll3dAxis1(b, -1)

    start = time.time()
    res0 = np.roll(b, 1, axis=1)
    mid = time.time()
    res1 = roll3dAxis1(b, 1)
    end = time.time()

    print("Native roll ran in {} and modified roll ran in {}".format(mid-start, end-mid))
    print(np.all(np.equal(res0, res1)))



