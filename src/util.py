import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy
from collections import namedtuple

Action = namedtuple('Action', ['position', 'action'])

Perspective = namedtuple('Perspective', ['perspective', 'position'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1

def load_network(PATH, device):
    model = torch.load(PATH, map_location='cpu')
    model = model.to(device)
    return model

def save_network(model, PATH):
    torch.save(model, PATH)

def pad_circular(x, pad):
    x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
    x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
    x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
    x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
    return x


def incrementalMean(x, mu, N):
    return mu + (x - mu) / (N)


def convert_from_np_to_tensor(tensor):
    tensor = from_numpy(tensor)
    tensor = tensor.type('torch.Tensor')
    return tensor
    
def generatePerspective(grid_shift, toric_size, state):
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
    def mod(index, shift):
        index = (index + shift) % toric_size 
        return index
    perspectives = []
    vertex_matrix = state[0,:,:]
    plaquette_matrix = state[1,:,:]
    # qubit matrix 0
    for i in range(toric_size):
        for j in range(toric_size):
            if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
            plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                temp = Perspective(new_state, (0,i,j))
                perspectives.append(temp)
    # qubit matrix 1
    for i in range(toric_size):
        for j in range(toric_size):
            if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
            plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                new_state = rotate_state(new_state) # rotate perspective clock wise
                temp = Perspective(new_state, (1,i,j))
                perspectives.append(temp)
    
    return perspectives

def rotate_state(state):
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        rot_plaquette_matrix = np.rot90(plaquette_matrix)
        rot_vertex_matrix = np.rot90(vertex_matrix)
        rot_vertex_matrix = np.roll(rot_vertex_matrix, 1, axis=0)
        rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
        return rot_state 

    
def shift_state(row, col, previous_state, state, grid_shift):
        previous_perspective = np.roll(previous_state, grid_shift-row, axis=1)
        previous_perspective = np.roll(previous_perspective, grid_shift-col, axis=2)
        perspective = np.roll(state, grid_shift-row, axis=1)
        perspective = np.roll(perspective, grid_shift-col, axis=2)
        return previous_perspective, perspective
