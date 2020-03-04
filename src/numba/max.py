import numpy as np
from numba import njit

@njit
def max2dAxis1(matrix):
    result = np.empty(matrix.shape[0])
    for row in range(matrix.shape[0]):
        result[row] = np.amax(matrix[row])
    return result.astype(matrix.dtype)

@njit
def max3dAxis2(cube):
    result = np.empty((cube.shape[0], cube.shape[1]))
    for mat in range(cube.shape[0]):
        result[mat] = max2dAxis1(cube[mat])
    return result.astype(cube.dtype)


import time
if __name__ == "__main__":
    mat = np.random.randint(0,100,(20, 500, 3))

    # compile calls
    max2dAxis1(mat[0])
    max3dAxis2(mat)


    start = time.time()
    res0 = np.amax(mat, axis=2)
    mid = time.time()
    res1 = max3dAxis2(mat)
    end = time.time()

    print(np.all(np.equal(res0, res1)))
    print("Native finshed in {}, Numba finished in {}".format(mid-start, end-mid))