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
from src.util_actor import selectAction, selectActionParallel, generateTransitionParallel, computePrioritiesParallel
from src.EnvSet import EnvSet
from src.util import action_type

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.nn.torch.ResNet import ResNet18

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
from src.util_actor import selectAction, computePriorities, generateTransition

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
            

def actor(rank, world_size, args):
     
            
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
    # NN = args["model"]
    # if NN == NN_11 or NN == NN_17:
    #     NN_config = args["model_config"]
    #     model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    # else:
    #     model = NN()
        
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
    print("REG")
    while True:
        
        steps_per_episode += 1    
        # if steps_per_episode > 1:
        #     return

        # select action using epsilon greedy policy
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"][0], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)

        next_state, reward, terminal_state, _ = env.step(action)
        
        # print(action)
        # print("REG\n", next_state)

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
            # print(transition)
            # print("REG\n", transition)
            local_buffer_T[buffer_idx] = transition
            local_buffer_Q[buffer_idx] = n_step_Q[n_step_idx-n_step]
            buffer_idx += 1

        n_step_idx = (n_step_idx+1) % n_step

        if buffer_idx >= size_local_memory_buffer:

            priorities = computePriorities( local_buffer_T[:-n_step],
                                            local_buffer_Q[:-n_step],
                                            np.roll(local_buffer_Q, -n_step, axis=0)[:-n_step],
                                            discount_factor**n_step)

            to_send = [*zip(local_buffer_T[:-n_step], priorities)]
            # print(to_send)

            return to_send
            # send buffer to learner
            # transition_queue_to_memory.put(to_send)
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
    



def actor_par(rank, world_size, args):

    no_envs = args["no_envs"]
    device = args["device"]
    discount_factor = args["discount_factor"]
    epsilon = np.array(args["epsilon"])

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
    # n_step                      = args["n_step"]
    size_local_memory_buffer    = args["size_local_memory_buffer"] + 1
    local_buffer_T              = np.empty((no_envs, size_local_memory_buffer), dtype=transition_type)    # Transtions
    local_buffer_A              = np.empty((no_envs, size_local_memory_buffer, 4), dtype=np.int) # A values
    local_buffer_Q              = np.empty((no_envs, size_local_memory_buffer), dtype=(np.float, 3)) # Q values
    local_buffer_R              = np.empty((no_envs, size_local_memory_buffer))    # R values
    buffer_idx                  = 0
    
    # # set network to eval mode
    # NN = args["model"]
    # if NN == NN_11 or NN == NN_17:
    #     NN_config = args["model_config"]
    #     model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    # else:
    #     model = NN()
        
    model.to(device)
    model.eval()
       
    # main loop over training steps
    print("PAR")
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
        # print(action)
        next_state, reward, terminal_state, _ = envs.step(action)
        # print(state, "\n")
        # print(next_state)

        transition = generateTransitionParallel(action,
                                                reward, 
                                                state,
                                                next_state, 
                                                terminal_state,
                                                grid_shift,
                                                transition_type)
        # print(transition)

        local_buffer_T[:, buffer_idx] = transition
        local_buffer_A[:, buffer_idx] = action
        local_buffer_Q[:, buffer_idx] = q_values
        local_buffer_R[:, buffer_idx] = reward
        buffer_idx += 1

        # If buffer full, send transitions
        if buffer_idx >= size_local_memory_buffer:
            print(local_buffer_Q)
            priorities = computePrioritiesParallel(local_buffer_A[:,:-1],
                                                   local_buffer_R[:,:-1],
                                                   local_buffer_Q[:,:-1],
                                                   np.roll(local_buffer_Q, -1, axis=1)[:,:-1],
                                                   discount_factor)

            to_send = [*zip(local_buffer_T[:,:-1].flatten(), priorities.flatten())]
            # print(to_send)

            return to_send

            # send buffer to learner
            # transition_queue_to_memory.put(to_send)
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

        # print(count)
    


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
        , "epsilon": [0.4,0.6,0.8]
        , "discount_factor" : 0.95
        , "max_actions_per_episode" : 75
        , "size_local_memory_buffer": 1
        , "n_step": 1
        , "no_envs": 3
    }

    # res    = actor(0,0, args)
    respar = actor_par(0,0, args)
    # print(respar)
    # print("")
    # print(res)

    # for i in range(len(respar)):
    #     nt = respar[i][0]
    #     npr = respar[i][1]
    #     t = res[i][0]
    #     pr = res[i][1]

    #     print(np.all(np.equal(t.state, nt[0])))
    #     print(t.reward == nt[2])
    #     print(np.all(np.equal(t.next_state, nt[3])))
    #     print(pr==npr)
    #     if not (np.all(np.equal(t.state, nt[0])) and t.reward == nt[2] and np.all(np.equal(t.next_state, nt[3]))):
    #         print("Actions: {}, {}".format(t.action, nt[1]))
    #         print("State: (Equal? {}) \n{}\n--------\n{}".format(np.all(np.equal(t.state, nt[0])),t.state, nt[0]))
    #         print("Reward: (Equal? {}) \n{}, {}".format(t.reward == nt[2],t.reward, nt[2]))
    #         print("Next State: (Equal? {}) \n{}\n--------\n{}".format(np.all(np.equal(t.next_state, nt[3])), t.next_state, nt[3]))
    #         print("Priorities: (Equal? {}) \n{}, {}".format(pr==npr,pr, npr))
    #         exit()

    for i in range(len(respar)):
        print(respar[i])