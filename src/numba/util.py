import numpy as np
from collections import namedtuple
from src.numba.roll import roll2dAxis0, roll2dAxis1, roll3dAxis1, roll3dAxis2
from src.numba.rotate import rot902d
from numba import njit


Action = namedtuple('Action', ['position', 'action'])
action_type = np.dtype([('position', (np.int, 3)), ('op', np.int)])

Perspective = namedtuple('Perspective', ['perspective', 'position'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])

@njit
def shift_state(row, col, previous_state, state, grid_shift):
    previous_perspective = roll3dAxis1(previous_state, grid_shift-row)
    previous_perspective = roll3dAxis2(previous_perspective, grid_shift-col)
    perspective          = roll3dAxis1(state, grid_shift-row)
    perspective          = roll3dAxis2(perspective, grid_shift-col)
    return previous_perspective, perspective


@njit
def rotate_state(state):
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        rot_plaquette_matrix = rot902d(plaquette_matrix)
        rot_vertex_matrix = rot902d(vertex_matrix)
        rot_vertex_matrix = roll2dAxis0(rot_vertex_matrix, 1)
        rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
        return rot_state 


@njit
def generatePerspectiveOptimized(grid_shift, toric_size, state):
    """ Generates the perspectives for a given syndrom.

    Params
    ======
    grid_shift: (int)
    toric_size: (int)
    state:      (np.ndarray)

    Return
    ======
    (np.ndarray) All the perspectives
    """
    perspectives = []
    positions    = []
    vm = state[0,:,:]
    pm = state[1,:,:]
    
    # qubit matrix 0
    vme = np.where(roll2dAxis0(vm, -1), 1, 0)
    pme = np.where(roll2dAxis1(pm,  1), 1, 0)
    err0 = np.logical_or(vm, vme)
    err1 = np.logical_or(pm, pme)
    err  = np.logical_or(err0, err1)
    args = np.argwhere(err == 1)
    for a in range(len(args)):
        i, j = args[a]
        new_state = roll3dAxis1(state, grid_shift-i)
        new_state = roll3dAxis2(new_state, grid_shift-j)
        perspectives.append(new_state)
        positions.append((0,i,j))

    # qubit matrix 1
    vme = np.where(roll2dAxis1(vm, -1), 1, 0)
    pme = np.where(roll2dAxis0(pm,  1), 1, 0)
    err0 = np.logical_or(vm, vme)
    err1 = np.logical_or(pm, pme)
    err  = np.logical_or(err0, err1)
    args = np.argwhere(err == 1)
    for a in range(len(args)):
        i, j = args[a]
        new_state = roll3dAxis1(state, grid_shift-i)
        new_state = roll3dAxis2(new_state, grid_shift-j)
        new_state = rotate_state(new_state) # rotate perspective clock wise
        perspectives.append(new_state)
        positions.append((1,i,j))

    return perspectives, positions

