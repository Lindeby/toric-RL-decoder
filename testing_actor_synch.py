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
from src.util_actor import updateRewards, selectAction, computePriorities, generateTransition
from src.EnvSet import EnvSet

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

    # # queues
    # con_learner = args["con_learner"]
    # transition_queue_to_memory = args["transition_queue_to_memory"] 

    no_envs = 2
    device = args["device"]
    discount_factor = args["discount_factor"]

    # local buffer of fixed size to store transitions before sending
    n_step                      = args["n_step"]
    size_local_memory_buffer    = args["size_local_memory_buffer"] + n_step
    local_buffer_T              = np.empty((no_envs, size_local_memory_buffer)) # Transtions
    local_buffer_Q              = np.empty((no_envs, size_local_memory_buffer)) # Q values
    buffer_idx                  = np.zeros(size_local_memory_buffer)
    n_step_S                    = np.empty((no_envs, n_step)) # State
    n_step_A                    = np.empty((no_envs, n_step)) # Actions
    n_step_Q                    = np.empty((no_envs, n_step)) # Q values
    n_step_R                    = np.zeros((no_envs, n_step)) # Rewards
    n_step_idx                  = np.zeros(size_local_memory_buffer) # index

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

    # env and env params
    env = gym.make(args["env"], config=args["env_config"])
    envs = EnvSet(env, no_envs)

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # startup
    state = envs.resetAll()
    steps_per_episode = 0
   
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
        # TODO: Epsilons is not a list
        # TODO: Rewrite learner to handle generatePerspectives without Perspective tuple
        action, q_values = selectAction(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)

        # Works until here

        next_state, reward, terminal_state, _ = envs.step(action)

        n_step_A[n_step_idx] = action
        n_step_S[n_step_idx] = state
        n_step_Q[n_step_idx] = q_values
        n_step_R[n_step_idx] = 0
        n_step_R = updateRewards(n_step_R, n_step_idx, reward, n_step, discount_factor)

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
            priorities = computePriorities( local_buffer_T[:-n_step],
                                            local_buffer_Q[:-n_step],
                                            np.roll(local_buffer_Q, -n_step)[:-n_step],
                                            discount_factor**n_step)      
            to_send = [*zip(local_buffer_T[:-n_step], priorities)]

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
        , "epsilon": 0
        , "discount_factor" : 0.95
        , "max_actions_per_episode" : 75
        , "size_local_memory_buffer": 100
        , "n_step": 1
    }

    actor(0,0, args)