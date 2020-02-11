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
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state

# for testing
from src.nn.torch.NN import NN_17



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
        , con_receive_weights:      (multiprocessing.Connection) connection
                                    where new weights are received
        , transition_queue:         (multiprocessing.Queue) SimpleQueue where
                                    transitions are sent to replay buffer
    }

    """
     
    # queues
    con_receive_weights = args["con_receive_weights"]
    transition_queue_to_memory = args["transition_queue_to_memory"] 
            
    device = args["device"]

    # local buffer of fixed size to store transitions before sending
    local_buffer = [None] * args["size_local_memory_buffer"]
    q_values_buffer = [None] * args["size_local_memory_buffer"]

    # set network to eval mode
    NN = args["model"]
    NN_config = args["model_config"]

    model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    model.to(device)
    model.eval()
    
    # env and env params
    env = gym.make(args["env"], config=args["env_config"])

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # Get initial network params
    weights = None
    while weights == None:
        print("recieve weights")
        weights = con_receive_weights.recv() # blocking op
        print("done receiving weights")

    # load weights
    vector_to_parameters(weights, model.parameters())
    # model.load_state_dict(weights)


    # init counters
    steps_counter = 0
    update_counter = 1
    local_memory_index = 0
    
    # startup
    state = env.reset()
    steps_per_episode = 0
    terminal_state = False
   
    # main loop over training steps
    for iteration in range(args["train_steps"]):

        steps_per_episode += 1
        previous_state = state
        
        # select action using epsilon greedy policy
        action, q_values = select_action(number_of_actions=no_actions,
                                        epsilon=args["epsilon"], 
                                        grid_shift=grid_shift,
                                        toric_size = env.system_size,
                                        state = state,
                                        model = model,
                                        device = device)


        state, reward, terminal_state, _ = env.step(action)
        # print(action)
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

        # if new weights are available, update network
        if con_receive_weights.poll():
            weights = con_receive_weights.recv()
            vector_to_parameters(weights, model.parameters())

        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
            

def select_action(number_of_actions, epsilon, grid_shift,
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
        perspective = row
        max_q_action = col + 1
        action = [  batch_position_actions[perspective][0],
                    batch_position_actions[perspective][1],
                    batch_position_actions[perspective][2],
                    max_q_action]
        q_values = q_values_table[perspective]

    # select random action
    else:
        random_perspective = random.randint(0, number_of_perspectives-1)
        random_action = random.randint(0, number_of_actions-1) +1
        action = [  batch_position_actions[random_perspective][0],
                    batch_position_actions[random_perspective][1],
                    batch_position_actions[random_perspective][2],
                    random_action]
        q_values = q_values_table[random_perspective]

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
                        previous_state,   #   Previous state before action
                        state,            #   Current state    
                        terminal_state    #   True/False
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


# def computePriorities(local_buffer, q_value_buffer, grid_shift, system_size, device, model, discount_factor):
#     """ Computes the absolute temporal difference value.

#     Parameters
#     ==========
#     local_buffer:        (list) local transitions from the actor
#     q_value_buffer:     (list) q values of taken steps
#     grid_shift:         (int) grid shift of the toric code
#     device:             (string) {"cpu", "cuda"}
#     model:              (torch.nn) model to compute the q value
#                         for the best action
#     discount_factor:    (float) discount future rewards

#     Returns
#     =======
#     (torch.Tensor) absolute TD error.
#     """

#     def toNetInput(batch, device):
#         batch_input = np.stack(batch, axis=0)
#         # from np to tensor
#         tensor = from_numpy(batch_input)
#         tensor = tensor.type('torch.Tensor')
#         return tensor.to(device)
    
#     transitions         = Transition(*zip(*local_buffer))
#     next_state_batch    = toNetInput(transitions.next_state, device) 
#     reward_batch        = toNetInput(transitions.reward, device)

#     max_q_value_buffer = []

#     for state in next_state_batch:
#         perspectives = generatePerspective(grid_shift, system_size, state)

#         #perspectives = Perspective(*zip(*perspectives))
#         #print(perspectives)
        
#         perspectives, positions = zip(*perspectives) 
#         batch_perspectives = np.array(perspectives)
#         #batch_perspectives = np.array(perspectives.perspective)
#         batch_perspectives = from_numpy(batch_perspectives).type('torch.Tensor').to(device)

#         with torch.no_grad():
#             output = model(batch_perspectives)
#             q_values_table = np.array(output.cpu())
#             row, col = np.unravel_index(np.argmax(q_values_table, axis=None), q_values_table.shape) 
#             perspective = row
#             max_q_action = col + 1
#             max_q_value = q_values_table[row, col]
#             max_q_value_buffer.append(max_q_value)
    
#     max_q_value_buffer  = toNetInput(np.array(max_q_value_buffer), device)
#     q_value_buffer      = toNetInput(np.array(q_value_buffer), device)
#     return torch.abs(reward_batch + discount_factor*max_q_value_buffer - q_value_buffer)
