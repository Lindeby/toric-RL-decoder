# torch
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

import gym, gym_ToricCode
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state
from src.actor import selectAction, generateTransition

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.nn.torch.ResNet import ResNet18

def actor(rank, world_size, args):
     
    device = args["device"]

    # local buffer of fixed size to store transitions before sending
    local_buffer = [None] * args["size_local_memory_buffer"]
    q_values_buffer = [None] * args["size_local_memory_buffer"]

    # set network to eval mode
    NN = args["model"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN
    
    model.to(device)
    model.eval()
    
    # env and env params
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # init counters
    steps_counter = 0
    update_counter = 1
    local_memory_index = 0
    
    # startup
    state1 = env.reset()
    steps_per_episode = 0
    terminal_state = False
   
    # main loop over training steps
    while True:
        # print(steps_counter)
        steps_counter += 1

        steps_per_episode += 1
        previous_state = state1
        
        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state1,
                                        model = model,
                                        device = device)


        state1, reward, terminal_state, _ = env.step(action)

        # generate transition to store in local memory buffer
        transition = generateTransition(action,
                                        reward,
                                        grid_shift,
                                        previous_state,
                                        state1,
                                        terminal_state)

        local_buffer[local_memory_index] = transition
        q_values_buffer[local_memory_index] = q_values
        local_memory_index += 1

        if (local_memory_index >= (args["size_local_memory_buffer"])): 
            priorities = computePriorities(local_buffer, q_values_buffer, args["discount_factor"])      
            to_send = [*zip(local_buffer, priorities)]

            # send buffer to learner
            return to_send
            local_memory_index = 0

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            state1 = env.reset()
            steps_per_episode = 0
            terminal_state = False
    

def actor_n_step(rank, world_size, args):
     
            
    device = args["device"]

    discount_factor = args["discount_factor"]

    # local buffer of fixed size to store transitions before sending
    n_step                      = args["n_step"]
    size_local_memory_buffer    = args["size_local_memory_buffer"] + n_step
    local_buffer_T              = [None] * size_local_memory_buffer # Transtions
    local_buffer_Q              = [None] * size_local_memory_buffer # Q values
    buffer_idx                  = 0
    n_step_S                    = [None] * n_step # State
    n_step_A                    = [None] * n_step # Actions
    n_step_Q                    = [None] * n_step # Q values
    n_step_R                    = [0   ] * n_step # Rewards
    n_step_idx                  = 0 # index

    # set network to eval mode
    NN = args["model"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN
        
    model.to(device)
    model.eval()
    
    # env and env params
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    

    # init counters
    steps_counter = 0
    update_counter = 1
    local_memory_index = 0
    
    # startup
    state = env.reset()
    steps_per_episode = 0
   
    # main loop over training steps
    while True:
        
        steps_per_episode += 1
        # previous_state = state1
        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=0, 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)

        next_state, reward, terminal_state, _ = env.step(action)

        n_step_A[n_step_idx] = action
        n_step_S[n_step_idx] = state # Keep in mind that next state is not saved until next iteration
        n_step_Q[n_step_idx] = q_values
        n_step_R[n_step_idx] = 0
        n_step_R = updateRewards(n_step_R, n_step_idx, reward, n_step, discount_factor) # Remember to 0 as well!

        if not (None in n_step_A):
            transition = generateTransition(n_step_A[n_step_idx-n_step],
                                            n_step_R[n_step_idx-n_step],
                                            grid_shift, 
                                            n_step_S[n_step_idx-n_step],
                                            next_state, 
                                            terminal_state        
                                            )
            local_buffer_T[buffer_idx] = transition
            local_buffer_Q[buffer_idx] = n_step_Q[n_step_idx-n_step]
            buffer_idx += 1

        n_step_idx = (n_step_idx+1) % n_step


        if buffer_idx >= size_local_memory_buffer:
            
            # disregard latest transition ssince it has no next state to compute priority for
            priorities = computePriorities_n_step(local_buffer_T[:-n_step],
                                                  local_buffer_Q[:-n_step],
                                                  np.roll(local_buffer_Q, -n_step)[:-n_step],
                                                  discount_factor**n_step)      
            to_send = [*zip(local_buffer_T[:-n_step], priorities)]

            # send buffer to learner
            return to_send
            buffer_idx = 0

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            # Reset n_step buffers
            n_step_S        = [None] * n_step
            n_step_A        = [None] * n_step
            n_step_R        = [0   ] * n_step

            # reset env
            state = env.reset()
            steps_per_episode = 0
        else:
            state = next_state
        


def updateRewards(reward_buffer, idx, reward, n_step, discount_factor):
    for t in range(0, n_step):
        reward_buffer[idx - t] += (discount_factor)**(t)*reward
    return reward_buffer


def computePriorities_n_step(local_buffer_trans, local_buffer_qs, local_buffer_qs_ns, discount_factor):
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

    transitions     = Transition(*zip(*local_buffer_trans))
    reward_batch    = np.array(transitions.reward)          # get rewards
    qv_max_ns       = np.amax(local_buffer_qs_ns, axis=1)   # get Qmax
    qv_st           = np.array([local_buffer_qs[i][a.action-1] for i, a in enumerate(transitions.action)])

    return np.absolute(reward_batch + discount_factor*qv_max_ns - qv_st)


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

    return np.absolute(reward_batch + discount_factor*q_values_max_next_state - q_values_state)


if __name__ == "__main__":
    model = ResNet18()
    env_config = {  
        "size": 3,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    args = {
        "model": model
        , "device": 'cuda' if torch.cuda.is_available() else 'cpu'
        , "env": 'toric-code-v0'
        , "env_config": env_config
        , "epsilon": 0
        , "discount_factor" : 0.95
        , "max_actions_per_episode" : 75
        , "size_local_memory_buffer": 100
        , "n_step": 1
    }

    n_step_trans = actor_n_step(0,0, args)
    one_step_trans = actor(0,0, args)

    for i in range(len(n_step_trans)):
        nt = n_step_trans[i][0]
        npr = n_step_trans[i][1]
        t = one_step_trans[i][0]
        pr = one_step_trans[i][1]

        if not (np.all(np.equal(t.state, nt.state)) and t.reward == nt.reward and np.all(np.equal(t.next_state, nt.next_state)) and pr==npr):
            print("Actions: {}, {}".format(t.action, nt.action))
            print("State: (Equal? {}) \n{}\n--------\n{}".format(np.all(np.equal(t.state, nt.state)),t.state, nt.state))
            print("Reward: (Equal? {}) \n{}, {}".format(t.reward == nt.reward,t.reward, nt.reward))
            print("Next State: (Equal? {}) \n{}\n--------\n{}".format(np.all(np.equal(t.next_state, nt.next_state)), t.next_state, nt.next_state))
            print("Priorities: (Equal? {}) \n{}, {}".format(pr==npr,pr, npr))
            exit()
        # print(not (np.all(np.equal(t.state, nt.state)) and t.reward == nt.reward and np.all(np.equal(t.next_state, nt.next_state)) and pr==npr))