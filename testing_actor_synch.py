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
from src.util_actor import updateRewards, selectAction, generateTransitionParallel, computePrioritiesParallel
from src.EnvSet import EnvSet
from src.util import action_type

# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.nn.torch.ResNet import ResNet18

            

def actor(rank, world_size, args):
    """ An actor that performs actions in an environment.

    Params
    ======
    rank:       (int) rank of the actor in a multiprocessing setting.
    world_size: (int) total number of actors and learners.
    args:       (dict) training specific parameters 
    {
        train_steps:                (int) number of training steps
        , max_actions_per_episode:  (int) number of actions before
                                    the episode is cut short
        , update_policy:            (int) (depricated) number of 
                                    steps until updating policy
        , size_local_memory_buffer: (int) size of the local replay buffer
        , min_qubit_errors:         (int) minumum number of qbit 
                                    errors on the toric code
        , model:                    (Class torch.nn) model to make predictions
        , model_config:             (dict)
        {
            system_size:        (int) size of the toric grid.
            , number_of_actions (int)
        }
        , env:                      (String) environment to act in
        , env_config                (dict)
        {
            size:               (int)
            , min_qbit_errors   (int)
            , p_error           (float)
        }
        , device:                   (String) {"cpu", "cuda"} device to
                                    operate whenever possible
        , epsilon:                  (float) probability of selecting a
                                    random action
        , beta:                     (float) parameter to determin the
                                    level of compensation for bias when
                                    computing priorities
        , n_step:                   (int) n-step learning
        , con_learner:              (multiprocessing.Connection) connection
                                    where new weights are received and termination
        , transition_queue:         (multiprocessing.Queue) SimpleQueue where
                                    transitions are sent to replay buffer
    }

    """
    

    '''
    no_envs       - int
    epsilon_dist - can be set, default linear fixed and for each actor
    
    '''
    no_envs = args["no_envs"]
    device = args["device"]
    discount_factor = args["discount_factor"]
    epsilon = np.array(args["epsilon"])

    # env and env params
    # TODO: Investigate if envs have the same mem adress
    env = gym.make(args["env"], config=args["env_config"])
    envs = EnvSet(env, no_envs)

    size = env.system_size

    transition_type = np.dtype([('perspective', (np.float, (2,size,size))),
                                ('action', action_type),
                                ('reward', np.float),
                                ('next_perspective', (np.float, (2,size,size))),
                                ('terminal',np.bool)])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(size/2)
    
    # startup
    state = envs.resetAll()
    steps_per_episode = np.zeros(no_envs)

    # # queues
    # con_learner = args["con_learner"]
    # transition_queue_to_memory = args["transition_queue_to_memory"] 



    # Local buffer of fixed size to store transitions before sending.
    n_step                      = args["n_step"]
    size_local_memory_buffer    = args["size_local_memory_buffer"] + n_step
    local_buffer_T              = np.empty((no_envs, size_local_memory_buffer), dtype=transition_type)    # Transtions
    local_buffer_A              = np.empty((no_envs, size_local_memory_buffer, 4), dtype=np.int) # A values
    local_buffer_Q              = np.empty((no_envs, size_local_memory_buffer), dtype=(np.float, 3)) # Q values
    local_buffer_R              = np.empty((no_envs, size_local_memory_buffer))    # R values
    buffer_idx                  = np.zeros( no_envs, dtype=np.int)
    n_step_S                    = np.empty((no_envs, n_step, 2, size, size))      # State
    n_step_A                    = np.empty((no_envs, n_step, 4), dtype=np.int)    # Actions
    n_step_Q                    = np.empty((no_envs, n_step, 3))    # Q values
    n_step_R                    = np.zeros((no_envs, n_step))       # Rewards
    n_step_idx                  = np.zeros( no_envs, dtype=np.int)  # Index
    n_step_full                 = np.zeros( no_envs, dtype=np.bool)



    # set network to eval mode
    NN = args["model"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN()
        
    model.to(device)
    model.eval()
    
    
    # # Get initial network params
    # weights = None
    # while weights == None:
    #     msg, weights = con_learner.recv() # blocking op
    #     if msg != "weights":
    #         weights = None

    # load weights
    # vector_to_parameters(weights, model.parameters())

    count = 0
   
    # main loop over training steps
    while True:
        
        # if con_learner.poll():
        #     msg, weights = con_learner.recv()
            
        #     if msg == "weights":
        #         vector_to_parameters(weights, model.parameters())
            
        #     elif msg == "prep_terminate":
        #         con_learner.send("ok")
        #         break

        steps_per_episode += 1

        # select action using epsilon greedy policy
        # TODO: Rewrite learner to handle generatePerspectives without Perspective tuple
        action, q_values = selectAction(number_of_actions=no_actions,
                                                epsilon=epsilon,
                                                grid_shift=grid_shift,
                                                toric_size = size,
                                                state = state,
                                                model = model,
                                                device = device)


        next_state, reward, terminal_state, _ = envs.step(action)

        n_step_A[:,n_step_idx] = action
        n_step_S[:,n_step_idx] = state
        n_step_Q[:,n_step_idx] = q_values
        n_step_R[  n_step_idx] = 0
        n_step_R = updateRewards(n_step_R, n_step_idx, reward, n_step, discount_factor)

        n_step_full_idx = np.argwhere(n_step_idx == (n_step-1)).flatten()

        if not (n_step_full_idx.size == 0):
            # a = n_step_A[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step]
            transition = generateTransitionParallel(n_step_A[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step],
                                                    n_step_R[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step], 
                                                    n_step_S[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step],
                                                    next_state[n_step_full_idx], 
                                                    terminal_state[n_step_full_idx],
                                                    grid_shift,
                                                    transition_type
                                                    )
            local_buffer_T[n_step_full_idx, buffer_idx[n_step_full_idx]] = transition
            local_buffer_A[n_step_full_idx, buffer_idx[n_step_full_idx]] = n_step_A[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step]
            local_buffer_Q[n_step_full_idx, buffer_idx[n_step_full_idx]] = n_step_Q[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step]
            local_buffer_R[n_step_full_idx, buffer_idx[n_step_full_idx]] = n_step_R[n_step_full_idx, n_step_idx[n_step_full_idx]-n_step]
            buffer_idx[n_step_full_idx] += 1

        n_step_idx  = (n_step_idx+1) % n_step

        # If buffer full, send transitions
        if np.max(buffer_idx) >= size_local_memory_buffer:
            # disregard latest transition since it has no next state to compute priority for
            priorities = computePrioritiesParallel(local_buffer_A,
                                                   local_buffer_R,
                                                   local_buffer_Q,
                                                   buffer_idx-n_step,
                                                   n_step,
                                                   discount_factor)

            to_send = [*zip(local_buffer_T[:,buffer_idx-n_step].flatten(), priorities)]

            exit()
            # to_send = [*zip(local_buffer_T[:-n_step], priorities)]

            # send buffer to learner
            # transition_queue_to_memory.put(to_send)
            buffer_idx = 0

        
        too_many_steps = steps_per_episode > args["max_actions_per_episode"]
        if np.any(terminal_state) or np.any(too_many_steps):
            
            # Reset terminal envs
            idx = np.argwhere(np.logical_or(terminal_state, too_many_steps)).flatten()
            reset_states = envs.resetTerminalEnvs(idx)

            # Reset n_step buffers
            n_step_idx[idx]  = 0
            n_step_full[idx] = False

            next_state[idx]        = reset_states
            steps_per_episode[idx] = 0
        
        state = next_state

        count += 1
        print(count)
    
    # ready to terminate
    # while True:
    #     msg, _ = con_learner.recv()
    #     if msg == "terminate":
    #         transition_queue_to_memory.close()
    #         con_learner.send("ok")
    #         break


if __name__ == "__main__":
    model = ResNet18
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
        , "epsilon": [0.3]
        , "discount_factor" : 0.95
        , "max_actions_per_episode" : 5
        , "size_local_memory_buffer": 1000
        , "n_step": 2
        , "no_envs": 15
    }

    actor(0,0, args)