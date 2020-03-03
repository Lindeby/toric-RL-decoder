import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

# torch
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

import gym
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util_actor import generateTransitionParallel, selectActionParallel, computePrioritiesParallel
from src.util import action_type
from src.EnvSet import EnvSet

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
            

def actor(rank, world_size, args):

    no_envs         = args["no_envs"]
    device          = args["device"]
    discount_factor = args["discount_factor"]
    epsilon         = np.array(args["epsilon"])

    # env and env params
    env = gym.make(args["env"], config=args["env_config"])
    envs = EnvSet(env, no_envs)

    size = env.system_size

    transition_type = np.dtype([('perspective', (np.int, (2,size,size))),
                                ('action', action_type),
                                ('reward', np.float),
                                ('next_perspective', (np.int, (2,size,size))),
                                ('terminal',np.bool)])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(size/2)
    
    # startup
    state = envs.resetAll()
    steps_per_episode = np.zeros(no_envs)


    # Local buffer of fixed size to store transitions before sending.
    size_local_memory_buffer    = args["size_local_memory_buffer"] + 1
    local_buffer_T              = np.empty((no_envs, size_local_memory_buffer), dtype=transition_type)  # Transtions
    local_buffer_A              = np.empty((no_envs, size_local_memory_buffer, 4), dtype=np.int)        # A values
    local_buffer_Q              = np.empty((no_envs, size_local_memory_buffer), dtype=(np.float, 3))    # Q values
    local_buffer_R              = np.empty((no_envs, size_local_memory_buffer))                         # R values
    buffer_idx                  = 0
    
    # set network to eval mode
    NN = args["model"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN()
        
    model.to(device)
    model.eval()
    count = 0
    # main loop over training steps
    while True:

        steps_per_episode += 1

        # select action using epsilon greedy policy
        action, q_values = selectActionParallel(number_of_actions=no_actions,
                                                epsilon=epsilon,
                                                grid_shift=grid_shift,
                                                toric_size = size,
                                                state = state,
                                                model = model,
                                                device = device)
        next_state, reward, terminal_state, _ = envs.step(action)

        transition = generateTransitionParallel(action,
                                                reward, 
                                                state,
                                                next_state, 
                                                terminal_state,
                                                grid_shift,
                                                transition_type)

        local_buffer_T[:, buffer_idx] = transition
        local_buffer_A[:, buffer_idx] = action
        local_buffer_Q[:, buffer_idx] = q_values
        local_buffer_R[:, buffer_idx] = reward
        buffer_idx += 1

        # If buffer full, send transitions
        if buffer_idx >= size_local_memory_buffer:
            priorities = computePrioritiesParallel(local_buffer_A[:,:-1],
                                                   local_buffer_R[:,:-1],
                                                   local_buffer_Q[:,:-1],
                                                   np.roll(local_buffer_Q, -1, axis=1)[:,:-1],
                                                   discount_factor)

            to_send = [*zip(local_buffer_T[:,:-1].flatten(), priorities.flatten())]

            return to_send

            buffer_idx = 0

        too_many_steps = steps_per_episode > args["max_actions_per_episode"]
        if np.any(terminal_state) or np.any(too_many_steps):
            
            # Reset terminal envs
            idx = np.argwhere(np.logical_or(terminal_state, too_many_steps)).flatten()
            reset_states = envs.resetTerminalEnvs(idx)

            # Reset n_step buffers
            next_state[idx]        = reset_states
            steps_per_episode[idx] = 0
        
        state = next_state
        count += 1
        # if count %10 == 0:
            # print(count)

from src.nn.torch.ResNet import ResNet18
import time, gym_ToricCode

if __name__ == "__main__":


    for i in range(1,100):
        TRANSITIONS_TO_GENERATE = 10000
        NO_ENVS = i

        env_config = {  
            "size": 3,
            "min_qubit_errors": 0,
            "p_error": 0.1
        }

        args = {
            "model": ResNet18
            , "device": 'cpu'
            , "env": 'toric-code-v0'
            , "env_config": env_config
            , "epsilon": [0.3]
            , "beta": 1
            , "discount_factor" : 0.95
            , "max_actions_per_episode" : 75
            , "n_step": 1
            , "no_envs": NO_ENVS
            , "size_local_memory_buffer": int(TRANSITIONS_TO_GENERATE/NO_ENVS)

        }

        # By doing like this, the compilation time for Numba will be included.
        # Run sufficiently many transitions so the compilation time becomes
        # negligeble
        start = time.time()
        actor(0,0, args)
        end = time.time()


        no_trans = int(TRANSITIONS_TO_GENERATE/NO_ENVS)*NO_ENVS
        trans_per_s = no_trans/(end-start)
        print("{} transitions in {} seconds using {} envs. ({} trans/s)".format(no_trans, end-start, i, trans_per_s))
        