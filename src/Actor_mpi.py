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
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state
from src.util_actor import updateRewards, selectAction, computePriorities, generateTransition

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
            

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
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # Get initial network params
    base_comm = args["mpi_base_comm"]
    learner_rank = args["mpi_learner_rank"]
    msg = None
    msg = base_comm.bcast(msg, root=learner_rank)
     
    msg, weights = msg
    if msg != "weights":
         weights = None

    # load weights
    vector_to_parameters(weights.to(device), model.parameters())

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
        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
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
            # receive new weights
            msg = base_comm.bcast(msg, root=learner_rank)
            msg, weights = msg
            if msg == "weights":
                vector_to_parameters(weights.to(device), model.parameters())
            elif msg == "terminate":
                break; 
            # disregard latest transition ssince it has no next state to compute priority for
            priorities = computePriorities( local_buffer_T[:-n_step],
                                            local_buffer_Q[:-n_step],
                                            np.roll(local_buffer_Q, -n_step)[:-n_step],
                                            discount_factor**n_step)      
            to_send = [*zip(local_buffer_T[:-n_step], priorities)]
            # send buffer to learner
            base_comm.gather(to_send, root=learner_rank)
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
    
