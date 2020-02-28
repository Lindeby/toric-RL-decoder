
import random, torch
import numpy as np
from src.util import generatePerspectiveOptimized, rotate_state, shift_state, Perspective, Transition, Action
from torch import from_numpy


def selectAction(number_of_actions, epsilon, grid_shift,
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
    perspectives, positions = generatePerspectiveParallel(grid_shift, toric_size, state)

    splice_idx = [len(p) for p in perspectives]
    splice_idx = np.cumsum(splice_idx)
    try:
        perspectives = np.concatenate(perspectives)
    except:
        dum = 0
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
    q_values        = np.split(q_values_table, splice_idx[:-1])

    actions, q_values = selectActionParallel(q_values, positions, greedy)
    return actions, q_values
    

def selectActionParallel(q_values, positions, greedy):
    actions = np.empty((len(q_values), 4), dtype=np.int)
    q_v     = np.empty((len(q_values), 3))

    for state in range(len(q_values)):
        if greedy[state]:
            # unravel_index not supported by numba
            # This can be done with np.where() instead
            p, a = np.unravel_index(np.argmax(q_values[state], axis=None), q_values[state].shape)
        else:
            p = np.random.randint(0, len(q_values[state]))
            a = np.random.randint(0, 3)

        q_v[state,:]  = q_values[state][p]
        actions[state,:] = [ positions[state][p][0],
                             positions[state][p][1],
                             positions[state][p][2],
                             a +1]

    return actions, q_v



def generatePerspectiveParallel(grid_shift, toric_size, states):
    per_result = []
    pos_result = []
    for i in range(len(states)):
        per, pos = generatePerspectiveOptimized(grid_shift, toric_size, states[i])
        per_result.append(per)
        pos_result.append(pos)

    return np.array(per_result), np.array(pos_result)


def generateTransition( action, 
                        reward, 
                        grid_shift,       
                        previous_state,   
                        state,                
                        terminal_state    
                        ):
    """ Generates a transition tuple to be stored in the replay memory.

    Params
    ======
    action:         (np.array)
    reward:         (float)
    grid_shift:     (int)
    previous_state: (np.ndarry)
    state:          (np.ndarray)
    terminal_state: (bool)

    Return
    ======
    (tuple) a tuple to be stored in the replay buffer.
    """

    qubit_matrix = action[0]
    row = action[1]
    col = action[2]
    add_operator = action[3]
    if qubit_matrix == 0:
        previous_perspective, perspective = shift_state(row, col, previous_state, state, grid_shift)
        action = Action((0, grid_shift, grid_shift), add_operator)
    elif qubit_matrix == 1:
        previous_perspective, perspective = shift_state(row, col, previous_state, state, grid_shift)
        previous_perspective = rotate_state(previous_perspective)
        perspective = rotate_state(perspective)
        action = Action((1, grid_shift, grid_shift), add_operator)
    return Transition(previous_perspective, action, reward, perspective, terminal_state)

def generateTransitionParallel( action, 
                                reward, 
                                previous_state,   
                                state,                
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
    state:          (np.ndarray)
    terminal_state: (bool)

    Return
    ======
    (tuple) a tuple to be stored in the replay buffer.
    """

    result = np.empty(state.shape[0], dtype=trans_type)

    for i in range(state.shape[0]):
        qm  = action[i][0]
        row = action[i][1]
        col = action[i][2]
        op  = action[i][3]
        if qm == 0:
            previous_perspective, perspective = shift_state(row, col, previous_state[i], state[i], grid_shift)
            a = Action((0, grid_shift, grid_shift), op)
        elif qm == 1:
            previous_perspective, perspective = shift_state(row, col, previous_state[i], state[i], grid_shift)
            previous_perspective = rotate_state(previous_perspective)
            perspective = rotate_state(perspective)
            a = Action((1, grid_shift, grid_shift), op)

        result[i] = Transition(previous_perspective, a, reward[i], perspective, terminal_state[i]) 
    return result


def updateRewards(reward_buffer, idx, reward, n_step, discount_factor):
    for t in range(0, n_step):
        reward_buffer[:, idx - t] += (discount_factor**t)*reward
    return reward_buffer


def computePriorities(local_buffer_trans, local_buffer_qs, local_buffer_qs_ns, discount_factor):
    """ Computes the absolute temporal difference value.

    Parameters
    ==========
    local_buffer:        (list) local transitions from the actor
    q_values_buffer:     (list) q values for respective action
    discount_factor:     (float) discount future rewards

    Returns
    =======
    (np.array) absolute TD error. (r + Qmax(st+1, a) - Q(st,a))
    """
    # TODO: This can be done a lot faster by simply storing rewards in an array as well.
    # Its a time consuming process trying to extraxt the rewards

    transitions     = Transition(*zip(*local_buffer_trans))
    reward_batch    = np.array(transitions.reward)          # get rewards
    qv_max_ns       = np.amax(local_buffer_qs_ns, axis=1)   # get Qmax
    qv_st           = np.array([local_buffer_qs[i][a.action-1] for i, a in enumerate(transitions.action)])

    return np.absolute(reward_batch + discount_factor*qv_max_ns - qv_st)



def computePrioritiesParallel(A,R,Q, idx, n_step, discount):
    """ Computes the absolute temporal difference value.

    Parameters
    ==========
    A:        (np.array)
    R:        (np.array)
    Q:        (np.array)
    idx:      (np.array)
    discount: (float)       Index for the final value to grab. Needed
                            due to A, R, Q can have different amount of
                            transitions stored in each environment.

    Returns
    =======
    (np.array) absolute TD error.
    """
    splices    = [0]
    for i, _ in enumerate(idx):
        splices.append(splices[i] + idx[i])

    # container for result
    priorities = np.empty(splices[-1])
    # Start and stop insert range for each env
    splices    = zip(splices, np.roll(splices,-1)[:-1])

    for env, (start, stop) in enumerate(splices):
        Qns = np.roll(Q[env], -n_step, axis=0)[:idx[env]] # Roll Q back to get Q next state
        qv_max_ns = np.amax(Qns, axis = 1)                # Find max Q value for next state
        actions   = A[env, :idx[env], -1] -1              # Take the actions we performed
        qv = Q[env, :idx[env]]
        qv = qv[np.arange(len(actions)), actions]         # Take Q values for actions we performed
        priorities[start:stop] = np.absolute(R[env, :idx[env]] + (discount**n_step)*qv_max_ns - qv)
    
    return priorities
