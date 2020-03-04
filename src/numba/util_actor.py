import random, torch
import numpy as np
from src.numba.util import generatePerspectiveOptimized, rotate_state, shift_state, Perspective, Transition, Action
from src.numba.max import max3dAxis2
from torch import from_numpy
from numba import njit, jit
from numba.typed import List


def selectActionParallel(number_of_actions, epsilon, grid_shift,
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
    perspectives, positions, splice_idx = generatePerspectiveParallel(grid_shift, toric_size, state)

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

    actions, q_values = selectActionParallel_prime(q_values_table, splice_idx, positions, greedy)
    return actions.astype('int64'), q_values


@njit
def generatePerspectiveParallel(grid_shift, toric_size, states):
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
def selectActionParallel_prime(q_values_table, splice_idx, positions, greedy):
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


def generateTransitionParallel( action, 
                                reward, 
                                state,   
                                next_state,                
                                terminal_state,
                                grid_shift,
                                trans_type
                                ):
    """ Generates a transition tuple to be stored in the replay memory.

    Params
    ======
    action:         (np.array)
    reward:         (float)
    grid_shift:     (int)
    previous_state: (np.ndarry)
    next_state:     (np.ndarray)
    terminal_state: (bool)

    Return
    ======
    (np.ndarray)
    """

    result = np.empty(next_state.shape[0], dtype=trans_type)

    for i in range(next_state.shape[0]):
        qm  = action[i][0]
        row = action[i][1]
        col = action[i][2]
        op  = action[i][3]
        if qm == 0:
            previous_perspective, perspective = shift_state(row, col, state[i], next_state[i], grid_shift)
            a = Action((0, grid_shift, grid_shift), op)
        elif qm == 1:
            previous_perspective, perspective = shift_state(row, col, state[i], next_state[i], grid_shift)
            previous_perspective = rotate_state(previous_perspective)
            perspective = rotate_state(perspective)
            a = Action((1, grid_shift, grid_shift), op)

        result[i] = Transition(previous_perspective, a, reward[i], perspective, terminal_state[i])
    return result



def computePrioritiesParallel(A,R,Q,Qns,discount):
    """ Computes the absolute temporal difference value.

    Parameters
    ==========
    A:        (np.array)
    R:        (np.array)
    Q:        (np.array)
    discount: (float)
    beta:     (float)

    Returns
    =======
    (np.array) absolute TD error.
    """
    Qns_max     = np.amax(Qns, axis=2)
    actions     = A[:,:,-1] -1
    row         = np.arange(actions.shape[-1])
    Qv          = np.array([Q[env,row,actions[env]] for env in range(len(Q))])
    return np.absolute(R + discount*Qns_max - Qv) 
