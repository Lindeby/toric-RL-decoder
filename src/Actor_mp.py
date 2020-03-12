# torch
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

# python lib
import numpy as np 
import gym, gym_ToricCode

# from file 
from src.util import action_type
from src.util_actor import generateTransitionParallel, computePrioritiesParallel
from src.numba.util_actor import selectActionBatch

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.EnvSet import EnvSet
import time            

def actor(args):
     
    device          = args["device"]
    discount_factor = args["discount_factor"]
    no_envs         = args["no_envs"]
    device          = args["device"]
    discount_factor = args["discount_factor"]
    epsilon_final   = np.array(args["epsilon_final"])
    epsilon         = np.ones(len(epsilon_final))
    epsilon_delta   = args["epsilon_delta"]

    # env and env params
    env  = gym.make(args["env"], config=args["env_config"])
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

    # Comms
    actor_io_queue         = args["actor_io_queue"] # Transition queue
    shared_mem_weights     = args["shared_mem_weights"]
    shared_mem_weight_id   = args["shared_mem_weight_id"]
    current_weight_id      = 0
    new_weights            = False

    # Init networkds
    NN              = args["model"]
    model_no_params = args["model_no_params"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN()
    
    # load initial network weights
    weights = np.empty(model_no_params)
    with shared_mem_weights.get_lock():
        reader = np.frombuffer(shared_mem_weights.get_obj())
        np.copyto(weights, reader)
    vector_to_parameters(from_numpy(weights).to(device).type(torch.FloatTensor), model.parameters())
    
    model.to(device)
    model.eval()
    
    preformence_start = time.time()
    preformence_stop = None
    print("Actor: starting loop.")
    # main loop over training steps
    while True:
        
        debug = time.time() 
        steps_per_episode += 1    
        
        # select action using epsilon greedy policy
        action, q_values = selectActionBatch(number_of_actions=no_actions,
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
            
           
             # read new weights from shared memory

            with shared_mem_weights.get_lock():
                if current_weight_id < shared_mem_weight_id.value:
                    reader = np.frombuffer(shared_mem_weights.get_obj())
                    np.copyto(weights, reader)
                    current_weight_id = shared_mem_weight_id.value
                    new_weights = True
            # load new weights into model
            if new_weights:
                new_weights = False
                vector_to_parameters(from_numpy(weights).type(torch.FloatTensor).to(device), model.parameters())
                epsilon = np.maximum(epsilon - epsilon_delta, epsilon_final)

            # compute priorities
            priorities = computePrioritiesParallel(local_buffer_A[:,:-1],
                                                   local_buffer_R[:,:-1],
                                                   local_buffer_Q[:,:-1],
                                                   np.roll(local_buffer_Q, -1, axis=1)[:,:-1],
                                                   discount_factor)

            to_send = [*zip(local_buffer_T[:,:-1].flatten(), priorities.flatten())]
            
            
            performence_stop = time.time()
            performence_elapsed = performence_stop - preformence_start
            performence_transitions = len(to_send)
            print("generating ",performence_transitions/performence_elapsed, "tranistions/s")
            preformence_start = time.time()

            # send buffer to learner
            actor_io_queue.put(to_send)
            buffer_idx = 0

        too_many_steps = steps_per_episode > args["max_actions_per_episode"]
        if np.any(terminal_state) or np.any(too_many_steps):
            
            # Reset terminal envs
            idx = np.argwhere(np.logical_or(terminal_state, too_many_steps)).flatten()
            reset_states = envs.resetTerminalEnvs(idx)
            next_state[idx]        = reset_states
            steps_per_episode[idx] = 0
        
        state = next_state

        print("debug time ",time.time()-debug)


    
