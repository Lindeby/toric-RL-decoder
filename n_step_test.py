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
from src.actor import selectAction, computePriorities, generateTransition, computePriorities_n_step

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
    state = env.reset()
    steps_per_episode = 0
    terminal_state = False
   
    # main loop over training steps
    while True:
        # print(steps_counter)
        steps_counter += 1

        steps_per_episode += 1
        previous_state = state
        
        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)


        state, reward, terminal_state, _ = env.step(action)

        # generate transition to store in local memory buffer
        transition = generateTransition(action,
                                        reward,
                                        grid_shift,
                                        previous_state,
                                        state,
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
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
    

def actor_n_step(rank, world_size, args):
     
            
    device = args["device"]

    discount_factor = args["discount_factor"]

    # N-step 
    n_step               = args["n_step"]
    n_step_idx           = 0
    n_step_state         = [None] * n_step
    n_step_action        = [None] * n_step
    n_step_reward        = [0   ] * n_step
    n_step_n_state       = [None] * n_step
    n_step_Qs_state      = [None] * n_step
    n_step_full          = False

    # local buffer of fixed size to store transitions before sending
    size_local_memory_buffer = args["size_local_memory_buffer"] + n_step
    local_buffer_trans  = [None] * size_local_memory_buffer
    local_buffer_qs     = [None] * size_local_memory_buffer
    local_buffer_qs_ns  = [None] * size_local_memory_buffer


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
        previous_state = state
        
        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)

        state, reward, terminal_state, _ = env.step(action)

        if n_step_full:
            a   = n_step_action[   n_step_idx - n_step]
            st  = n_step_state[    n_step_idx - n_step]
            r   = n_step_reward[   n_step_idx - n_step]
            # generate transition to store in local memory buffer
            transition = generateTransition(a, r, grid_shift, st, previous_state, terminal_state)
            local_buffer_trans[ local_memory_index]  = transition
            local_buffer_qs[    local_memory_index]  = n_step_Qs_state[n_step_idx - n_step]
            local_buffer_qs_ns[ local_memory_index]  = q_values
            local_memory_index += 1

        # n-step
        n_step_state[    n_step_idx ] = previous_state
        n_step_reward[   n_step_idx ] = 0
        n_step_action[   n_step_idx ] = action
        #n_step_n_state[  n_step_idx ] = state # wrong? Not used for now so might not matter
        n_step_Qs_state[ n_step_idx ] = q_values
        for n in range(1,n_step+1, 1):
            n_step_reward[n_step_idx - n] += (discount_factor**(n-1))*reward
        n_step_idx = 0 if n_step_idx >= n_step-1 else n_step_idx +1

        if n_step_idx >= n_step-1:
            n_step_full = True

        if (local_memory_index >= (args["size_local_memory_buffer"])): 
            # disregard lates transition since it has no next state to compute priority for
            priorities = computePriorities_n_step(local_buffer_trans[:-n_step], local_buffer_qs[:-n_step], local_buffer_qs_ns[:-n_step], discount_factor**n_step)      
            to_send = [*zip(local_buffer_trans, priorities)]

            # send buffer to learner
            return to_send
            local_memory_index = 0

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            # Reset n_step buffers
            n_step_state         = [None] * n_step
            n_step_action        = [None] * n_step
            n_step_reward        = [0   ] * n_step
            n_step_n_state       = [None] * n_step
            n_step_Qs_state      = [None] * n_step
            n_step_full          = False

            # reset env
            state = env.reset()
            steps_per_episode = 0


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
        , "size_local_memory_buffer": 10
        , "n_step": 1
    }

    n_step_trans = actor(0,0, args)
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

        print(not (np.all(np.equal(t.state, nt.state)) and t.reward == nt.reward and np.all(np.equal(t.next_state, nt.next_state)) and pr==npr))