# torch
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

# import gym
# from gym_ToricCode import ToricCode
from src.ToricCode import ToricCode
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state

# for testing
from src.nn.torch.NN import NN_17

from numpy import save

from pathlib import Path
import sys
import time
from torch.multiprocessing import Process

import objgraph

def actor(num, num_transitions):
    
    print("hello ", num)
            
    device = "cpu"
    generate_transitions = num_transitions #int(sys.argv[1])
    
    size = 9

    # set network to eval mode
    NN = NN_17
    model = NN(size, 3, device)
    model.eval()
    
    # env and env params
     
    env_config = {  "size": size,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
                 }
    # env = gym.make("toric-code-v0", config = env_config)

    env = ToricCode(env_config)

    no_actions = int(3)
    grid_shift = int(env.system_size/2)
    

    memory_transitions = []
    memory_q_values = []

    
    # startup
    state = env.reset()
    steps_per_episode = 0
    terminal_state = False
   
    print("Env size is {}".format(sys.getsizeof(env)))

    time_start = time.time()
    # main loop over training steps
    for iteration in range(generate_transitions):
        print(iteration)

        steps_per_episode += 1
        previous_state = state
        
        # select action using epsilon greedy policy
        action, q_values = select_action(number_of_actions=no_actions,
                                        epsilon=0.3, 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)

        state, reward, terminal_state, _ = env.step(action)

        print(action)
        # generate transition to store in local memory buffer
        # transition = generateTransition(action,
        #                                 reward,
        #                                 grid_shift,
        #                                 previous_state,
        #                                 state,
        #                                 terminal_state)
        
        # memory_transitions.append(transition)
        # memory_q_values.append(q_values)



        if terminal_state or steps_per_episode > 5:
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
    
    time_stop = time.time()
    time_elapsed = time_stop - time_start
    print("generated ",generate_transitions," transitions in ",time_elapsed) 
    save_name = 'output_speed_test/transitions_'+str(num)+'.npy'
    save_name_q = 'output_speed_test/q_values_'+str(num)+'.npy'
    Path("output_speed_test").mkdir(parents=True, exist_ok=True)
    save(save_name, memory_transitions)
    save(save_name_q,memory_q_values)
    #save('output_speed_test/transitions.npy', memory_transitions)
    #save('output_speed_test/q_values.npy', memory_q_values)
    
            
            

def select_action(number_of_actions, epsilon, grid_shift,
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
    perspectives = generatePerspective(grid_shift, toric_size, state)
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
        perspective = row
        max_q_action = col + 1
        action = [  batch_position_actions[perspective][0],
                    batch_position_actions[perspective][1],
                    batch_position_actions[perspective][2],
                    max_q_action]
        q_values = q_values_table[perspective]

    # select random action
    else:
        random_perspective = random.randint(0, number_of_perspectives-1)
        random_action = random.randint(0, number_of_actions-1) +1
        action = [  batch_position_actions[random_perspective][0],
                    batch_position_actions[random_perspective][1],
                    batch_position_actions[random_perspective][2],
                    random_action]
        q_values = q_values_table[random_perspective]

    return action, q_values
    

def computePriorities(local_buffer, q_values_buffer, discount_factor):
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

    transitions             = Transition(*zip(*local_buffer))
    reward_batch            = np.array(transitions.reward)
    q_values_max_next_state = np.amax(np.roll(q_values_buffer, -1), axis=1)
    q_values_state          = np.array([q_values_buffer[i][a.action-1] for i, a in enumerate(transitions.action)])

    return np.absolute(reward_batch - discount_factor*q_values_max_next_state - q_values_state)


def generateTransition( action, 
                        reward, 
                        grid_shift,       
                        previous_state,   #   Previous state before action
                        state,            #   Current state    
                        terminal_state    #   True/False
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


if __name__ == '__main__':
        
    num_actors = 1 #int(sys.argv[1])
    num_transitions = 50# int(sys.argv[2])
    #num_actors = 2
    
    actor(0, num_transitions)

    # for i in range(num_actors):
    #     a = Process(target = actor, args=(i, num_transitions))
    #     a.start()

#actor()

# def computePriorities(local_buffer, q_value_buffer, grid_shift, system_size, device, model, discount_factor):
#     """ Computes the absolute temporal difference value.

#     Parameters
#     ==========
#     local_buffer:        (list) local transitions from the actor
#     q_value_buffer:     (list) q values of taken steps
#     grid_shift:         (int) grid shift of the toric code
#     device:             (string) {"cpu", "cuda"}
#     model:              (torch.nn) model to compute the q value
#                         for the best action
#     discount_factor:    (float) discount future rewards

#     Returns
#     =======
#     (torch.Tensor) absolute TD error.
#     """

#     def toNetInput(batch, device):
#         batch_input = np.stack(batch, axis=0)
#         # from np to tensor
#         tensor = from_numpy(batch_input)
#         tensor = tensor.type('torch.Tensor')
#         return tensor.to(device)
    
#     transitions         = Transition(*zip(*local_buffer))
#     next_state_batch    = toNetInput(transitions.next_state, device) 
#     reward_batch        = toNetInput(transitions.reward, device)

#     max_q_value_buffer = []

#     for state in next_state_batch:
#         perspectives = generatePerspective(grid_shift, system_size, state)

#         #perspectives = Perspective(*zip(*perspectives))
#         #print(perspectives)
        
#         perspectives, positions = zip(*perspectives) 
#         batch_perspectives = np.array(perspectives)
#         #batch_perspectives = np.array(perspectives.perspective)
#         batch_perspectives = from_numpy(batch_perspectives).type('torch.Tensor').to(device)

#         with torch.no_grad():
#             output = model(batch_perspectives)
#             q_values_table = np.array(output.cpu())
#             row, col = np.unravel_index(np.argmax(q_values_table, axis=None), q_values_table.shape) 
#             perspective = row
#             max_q_action = col + 1
#             max_q_value = q_values_table[row, col]
#             max_q_value_buffer.append(max_q_value)
    
#     max_q_value_buffer  = toNetInput(np.array(max_q_value_buffer), device)
#     q_value_buffer      = toNetInput(np.array(q_value_buffer), device)
#     return torch.abs(reward_batch + discount_factor*max_q_value_buffer - q_value_buffer)
