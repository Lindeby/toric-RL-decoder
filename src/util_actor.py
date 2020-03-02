
import random, torch
import numpy as np
<<<<<<< .merge_file_wQS7qS
from src.util import generatePerspectiveOptimized, generatePerspective, rotate_state, shift_state, Perspective, Transition, Action
=======
from src.util import generatePerspectiveOptimized, rotate_state, shift_state, Perspective, Transition, Action
>>>>>>> .merge_file_TzGt0U
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
    perspectives = generatePerspectiveOptimized(grid_shift, toric_size, state)
    number_of_perspectives = len(perspectives)

    # preprocess batch of perspectives and actions
    perspectives = Perspective(*zip(*perspectives))
    batch_perspectives = np.array(perspectives.perspective)
    batch_perspectives = from_numpy(batch_perspectives).type('torch.Tensor')    
    batch_perspectives = batch_perspectives.to(device)
    batch_position_actions = perspectives.position
   
    # Policy
    policy_net_output = None
    q_values_table = None
    with torch.no_grad():
        policy_net_output = model(batch_perspectives)
        q_values_table = np.array(policy_net_output.cpu())


    #choose action using epsilon greedy approach
    rand = random.random()
    if(1 - epsilon > rand):
        # select greedy action
        row, col = np.unravel_index(np.argmax(q_values_table, axis=None), q_values_table.shape) 
        p = row
        a = col + 1

    # select random action
    else:
        p = random.randint(0, number_of_perspectives-1)
        a = random.randint(0, number_of_actions-1) +1

    action = [  batch_position_actions[p][0],
                batch_position_actions[p][1],
                batch_position_actions[p][2],
                a]
    q_values = q_values_table[p]
    return action, q_values


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
    perspectives, positions = generatePerspectiveParallel(grid_shift, toric_size, state)

    splice_idx = [len(p) for p in perspectives]
    splice_idx = np.cumsum(splice_idx)

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
    q_values        = np.split(q_values_table, splice_idx[:-1])

    actions, q_values = selectActionParallel_prime(q_values, positions, greedy)
    return actions, q_values


def generatePerspectiveParallel(grid_shift, toric_size, states):
    per_result = []
    pos_result = []
    for i in range(len(states)):
        per, pos = generatePerspectiveOptimized(grid_shift, toric_size, states[i])
        per_result.append(per)
        pos_result.append(pos)

    return np.array(per_result), np.array(pos_result)


def selectActionParallel_prime(q_values, positions, greedy):
    actions = np.empty((len(q_values), 4), dtype=np.int)
    q_v     = np.empty((len(q_values), 3))

    for state in range(len(q_values)):
        if greedy[state]:
            p, a = np.where(q_values[state] == np.max(q_values[state]))
            p = p[0]
            a = a[0]
        else:
            p = np.random.randint(0, len(q_values[state]))
            a = np.random.randint(0, 3)

        q_v[state,:]  = q_values[state][p]
        actions[state,:] = [ positions[state][p][0],
                             positions[state][p][1],
                             positions[state][p][2],
                             a +1]

    return actions, q_v


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
    (tuple) a tuple to be stored in the replay buffer.
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



def computePrioritiesParallel(A,R,Q,Qns,discount):
    """ Computes the absolute temporal difference value.

    Parameters
    ==========
    A:        (np.array)
    R:        (np.array)
    Q:        (np.array)
    discount: (float)       Index for the final value to grab. Needed
                            due to A, R, Q can have different amount of
                            transitions stored in each environment.

    Returns
    =======
    (np.array) absolute TD error.
    """
    Qns_max = np.amax(Qns, axis=2)
    actions = A[:,:,-1] -1
    row = np.arange(actions.shape[-1])
    Qv      = np.array([Q[env,row,actions[env]] for env in range(len(Q))])
    return np.absolute(R + discount*Qns_max - Qv)
