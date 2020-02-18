# torch
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

# import gym
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state


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
        , con_learner:              (multiprocessing.Connection) connection
                                    where new weights are received and termination
        , transition_queue:         (multiprocessing.Queue) SimpleQueue where
                                    transitions are sent to replay buffer
    }

    """
     
    # queues
    con_learner = args["con_learner"]
    transition_queue_to_memory = args["transition_queue_to_memory"] 
            
    device = args["device"]

    # local buffer of fixed size to store transitions before sending
    local_buffer = [None] * args["size_local_memory_buffer"]
    q_values_buffer = [None] * args["size_local_memory_buffer"]

    # set network to eval mode
    NN = args["model"]
    NN_config = args["model_config"]

    # model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    model = NN()
    model.to(device)
    model.eval()
    
    # env and env params
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # Get initial network params
    weights = None
    while weights == None:
        msg, weights = con_learner.recv() # blocking op
        if msg != "weights":
            weights = None

    # load weights
    vector_to_parameters(weights, model.parameters())


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
        # steps_counter += 1
        if con_learner.poll():
            msg, weights = con_learner.recv()
            
            if msg == "weights":
                vector_to_parameters(weights, model.parameters())
            
            elif msg == "prep_terminate":
                con_learner.send("ok")
                break

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

        if (local_memory_index >= (args["size_local_memory_buffer"]-1)): 
            # disregard lates transition since it has no next state to compute priority for
            priorities = computePriorities(local_buffer[0:-1], q_values_buffer[0:-1], args["discount_factor"])      
            to_send = [*zip(local_buffer, priorities)]

            # send buffer to learner
            transition_queue_to_memory.put(to_send)
            local_memory_index = 0
        else:
            local_memory_index += 1

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
    
    # ready to terminate
    while True:
        msg, _ = con_learner.recv()
        if msg == "terminate":
            transition_queue_to_memory.close()
            con_learner.send("ok")
            break
            
            

def selectAction(number_of_actions, epsilon, grid_shift,
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
        p = row
        a = col + 1

    # select random action
    else:
        p = random.randint(0, number_of_perspectives-1)
        a = random.randint(0, number_of_actions-1) +1

    action = [  batch_position_actions[p][0],
                batch_position_actions[p][1],
                batch_position_actions[p][2],
                a]
    q_values = q_values_table[p]
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
                        previous_state,   
                        state,                
                        terminal_state    
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





def actor_n_step(rank, world_size, args):
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
        , con_learner:              (multiprocessing.Connection) connection
                                    where new weights are received and termination
        , transition_queue:         (multiprocessing.Queue) SimpleQueue where
                                    transitions are sent to replay buffer
    }

    """
     
    # queues
    con_learner = args["con_learner"]
    transition_queue_to_memory = args["transition_queue_to_memory"] 
            
    device = args["device"]

    discount_factor = args["discount_factor"]

    # local buffer of fixed size to store transitions before sending
    local_buffer = [None] * args["size_local_memory_buffer"]
    q_values_buffer = [None] * args["size_local_memory_buffer"]

    # N-step 
    n_step      = 3
    n_step_idx  = 0
    n_step_state         = [None] * n_step
    n_step_action        = [None] * n_step
    n_step_reward        = [None] * n_step
    n_step_discount      = [None] * n_step
    n_step_n_state       = [None] * n_step
    n_step_Qs_state      = [None] * n_step
    n_step_Qs_n_state    = [None] * n_step


    # set network to eval mode
    NN = args["model"]
    NN_config = args["model_config"]

    # model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    model = NN()
    model.to(device)
    model.eval()
    
    # env and env params
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # Get initial network params
    weights = None
    while weights == None:
        msg, weights = con_learner.recv() # blocking op
        if msg != "weights":
            weights = None

    # load weights
    vector_to_parameters(weights, model.parameters())


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
        
        if con_learner.poll():
            msg, weights = con_learner.recv()
            
            if msg == "weights":
                vector_to_parameters(weights, model.parameters())
            
            elif msg == "prep_terminate":
                con_learner.send("ok")
                break

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

        # n-step
        # TODO: Think about terminal state
        n_step_state[       n_step_idx          ] = previous_state
        n_step_action[      n_step_idx          ] = action
        n_step_n_state[     n_step_idx          ] = state
        n_step_Qs_state[    n_step_idx          ] = q_values
        n_step_Qs_n_state[  n_step_idx - n_step ] = q_values
        for n in range(n_step):
            n_step_reward[  n_step_idx-n-1] += (discount_factor**n)*reward
            n_step_discount[n_step_idx-n-1 ] = discount_factor**n

        if not None in n_step_state:
            # generate transition for t-3 and store to local buffer
            # generate transition to store in local memory buffer
            transition = generateTransition(action,
                                            reward,
                                            grid_shift,
                                            previous_state,
                                            state,
                                            terminal_state)
            # local_buffer[local_memory_index] = transition
            # q_values_buffer[local_memory_index] = q_values
            pass



        if (local_memory_index >= (args["size_local_memory_buffer"]-1)): 
            # disregard lates transition since it has no next state to compute priority for
            priorities = computePriorities(local_buffer[0:-1], q_values_buffer[0:-1], args["discount_factor"])      
            to_send = [*zip(local_buffer, priorities)]

            # send buffer to learner
            transition_queue_to_memory.put(to_send)
            local_memory_index = 0
        else:
            local_memory_index += 1

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            # TODO: Clear n-step buffers
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
    
    # ready to terminate
    while True:
        msg, _ = con_learner.recv()
        if msg == "terminate":
            transition_queue_to_memory.close()
            con_learner.send("ok")
            break
       