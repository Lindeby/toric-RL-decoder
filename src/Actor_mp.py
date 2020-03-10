# torch
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

import gym, gym_ToricCode
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util import action_type
from src.util_actor import generateTransitionParallel, selectActionParallel, computePrioritiesParallel #,selectAction, computePriorities, generateTransition

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.EnvSet import EnvSet
            

def actor(args):
     
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
        model = NN()
        
    model.to(device)
    model.eval()
    
    # env and env params
    no_envs = args["no_envs"]
    env = gym.make(args["env"], config=args["env_config"])
    envs = EnvSet(env, no_envs)
    size = env.system_size

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    epsilon = args["epsilon"]
    
    
    transition_type = np.dtype([('perspective', (np.int, (2,size,size))),
                                ('action', action_type),
                                ('reward', np.float),
                                ('next_perspective', (np.int, (2,size,size))),
                                ('terminal',np.bool)])

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

    actor_io_queue = args["actor_io_queue"]
    pipe_actor_io = args["pipe_actor_io"]
    
    # Get initial network params
    weights = None
    while True:
        msg, w = pipe_actor_io.recv()
        if msg == "weights":
           weights = w
           pipe_actor_io.send("ok")
           break
        print("Actor recived initial weights") 
    
    # load weights
    vector_to_parameters(weights.to(device), model.parameters())

    # init counters
    steps_counter = 0
    update_counter = 1
    local_memory_index = 0
    # main loop over training steps
    
    print("Actor start loop")
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
            print("Actor Sync")
            # receive new weights
            if pipe_actor_io.poll():
                msg, weights = pipe_actor_io.recv()
                print(msg)
                if msg == "weights":
                    print("received weights")
                    vector_to_parameters(weights.to(device), model.parameters())
                    pipe_actor_io.send("ok")
                elif msg == "prep_terminate":
                    break; 

            priorities = computePrioritiesParallel(local_buffer_A[:,:-1],
                                                   local_buffer_R[:,:-1],
                                                   local_buffer_Q[:,:-1],
                                                   np.roll(local_buffer_Q, -1, axis=1)[:,:-1],
                                                   discount_factor)

            to_send = [*zip(local_buffer_T[:,:-1].flatten(), priorities.flatten())]

            # send buffer to learner
            actor_io_queue.put(to_send)
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
    
    print("actor send acc")
    pipe_actor_io.send("ok")
    
    while True:
        msg, _ = pipe_actor_io.recv()
        print(msg)
        if msg == "terminate":
            break
    pipe_actor_io.send("ok")
    print("actor done")


    
