import random, torch
import numpy as np
from src.numba.util import generatePerspectiveOptimized, rotate_state, shift_state
from src.util import Perspective, Transition, Action 
from src.numba.max import max3dAxis2
from torch import from_numpy
from numba import njit, jit
from numba.typed import List


def selectActionBatch(number_of_actions, epsilon, grid_shift,
                    toric_size, state, model, device):
    """ Selects an action according to a epsilon-greedy policy.

    Params
    ======
    number_actions: (int)
    epsilon:        (float)
    grid_shift:     (int)
    toric_size:     (int)
    state:          (np.ndarray)
    model:          (torch.nn)
    device:         (String) {"cpu", "cuda"}

    Return
    ======
    (tuple(np.array, float)) selected action and its q_value
    """
    # set network in evluation mode 
    model.eval()

    # generate perspectives
    perspectives, positions, splice_idx = generatePerspectiveBatch(grid_shift, toric_size, state)

    splice_idx = np.cumsum(splice_idx)
 
    positions    = np.concatenate(positions)
    perspectives = np.concatenate(perspectives)
    perspectives = from_numpy(perspectives).type('torch.Tensor').to(device)

    # Policy
    policy_net_output = None
    q_values_table = None
    with torch.no_grad():
        policy_net_output = model(perspectives)
        q_values_table = np.array(policy_net_output.cpu())

    #choose action using epsilon greedy approach
    rand            = np.random.random(len(state))
    greedy          = (1 - epsilon) > rand

    actions, q_values = _selectActionBatch_prime(q_values_table, splice_idx, positions, greedy)
    return actions.astype('int64'), q_values


@njit
def generatePerspectiveBatch(grid_shift, toric_size, states):
    per_result = List()
    pos_result = List()
    pos_idx    = List()
    for i in range(len(states)):
        per, pos = generatePerspectiveOptimized(grid_shift, toric_size, states[i])
        per_result.append(per)
        pos_result.append(pos)
        pos_idx.append(len(per))

    return per_result, pos_result, pos_idx

@njit
def _selectActionBatch_prime(q_values_table, splice_idx, positions, greedy):
    """ Helper function for selectActionParallel. This is the parallelizable part.

    Params
    ======
    q_values:   (np.ndarray)
    positions:  (np.ndarray)
    greedy:     (np.ndarray)

    Return
    ======
    (np.ndarray), (np.ndarray)
    """

    actions = np.empty((len(splice_idx), 4))
    q_v     = np.empty((len(splice_idx), 3))
    first = 0
    for i in range(len(splice_idx)):
        end = splice_idx[i]
        q_values = q_values_table[first:end]
        pos = positions[first:end]

        if greedy[i]:
            p0, a0 = np.where(q_values == np.amax(q_values))
            p = p0[0]
            a = a0[0]
        else:
            p = np.random.randint(0, len(q_values))
            a = np.random.randint(0, 3)

        q_v[i,:]     = q_values[p]
        actions[i,:] = [ pos[p][0],
                         pos[p][1],
                         pos[p][2],
                             a +1]
        first = splice_idx[i]

    return actions, q_v

