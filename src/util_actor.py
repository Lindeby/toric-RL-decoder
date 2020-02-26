
import random, torch
import numpy as np
from src.util import generatePerspectiveOptimized, rotate_state, shift_state, Perspective, Transition, Action
from torch import from_numpy


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
    perspectives, positions = generatePerspectiveParallel(grid_shift, toric_size, state)

    splice_idx = [len(p) for p in perspectives]
    perspectives = np.concatenate(perspectives)
    perspectives = from_numpy(perspectives).type('torch.Tensor').to(device)

    # Policy
    policy_net_output = None
    q_values_table = None
    with torch.no_grad():
        policy_net_output = model(perspectives)
        q_values_table = np.array(policy_net_output.cpu())

    #choose action using epsilon greedy approach
    rand            = np.random.random(len(state))
    greedy          = (1 - epsilon) > rand
    q_values        = np.split(q_values_table, splice_idx[:-1])

    actions, q_values = selectActionParallel(q_values, positions, greedy)
    return actions, q_values
    

def selectActionParallel(q_values, positions, greedy):
    actions = np.empty((len(q_values), 4))
    q_v     = np.empty((len(q_values), 3))

    for state in range(len(q_values)):
        if greedy[state]:
            p, a = np.unravel_index(np.argmax(q_values[state], axis=None), q_values[state].shape)
        else:
            p = np.randint(0, len(q_values[state]))
            a = np.randint(0, 3)

        q_v[state,:]  = q_values[state][p]
        actions[state,:] = [ positions[state][p][0],
                             positions[state][p][1],
                             positions[state][p][2],
                             a +1]

    return actions, q_v



def generatePerspectiveParallel(grid_shift, toric_size, states):
    per_result = []
    pos_result = []
    for i in range(len(states)):
        per, pos = generatePerspectiveOptimized(grid_shift, toric_size, states[i])
        per_result.append(per)
        pos_result.append(pos)

    return np.array(per_result), np.array(pos_result)


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


def updateRewards(reward_buffer, idx, reward, n_step, discount_factor):
    for t in range(0, n_step):
        reward_buffer[idx - t] += (discount_factor)**(t)*reward
    return reward_buffer


def computePriorities(local_buffer_trans, local_buffer_qs, local_buffer_qs_ns, discount_factor):
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

    transitions     = Transition(*zip(*local_buffer_trans))
    reward_batch    = np.array(transitions.reward)          # get rewards
    qv_max_ns       = np.amax(local_buffer_qs_ns, axis=1)   # get Qmax
    qv_st           = np.array([local_buffer_qs[i][a.action-1] for i, a in enumerate(transitions.action)])

    return np.absolute(reward_batch + discount_factor*qv_max_ns - qv_st)


# def actorOld(rank, world_size, args):
#     """ An actor that performs actions in an environment.

#     Params
#     ======
#     rank:       (int) rank of the actor in a multiprocessing setting.
#     world_size: (int) total number of actors and learners.
#     args:       (dict) training specific parameters 
#     {
#         train_steps:                (int) number of training steps
#         , max_actions_per_episode:  (int) number of actions before
#                                     the episode is cut short
#         , update_policy:            (int) (depricated) number of 
#                                     steps until updating policy
#         , size_local_memory_buffer: (int) size of the local replay buffer
#         , min_qubit_errors:         (int) minumum number of qbit 
#                                     errors on the toric code
#         , model:                    (Class torch.nn) model to make predictions
#         , model_config:             (dict)
#         {
#             system_size:        (int) size of the toric grid.
#             , number_of_actions (int)
#         }
#         , env:                      (String) environment to act in
#         , env_config                (dict)
#         {
#             size:               (int)
#             , min_qbit_errors   (int)
#             , p_error           (float)
#         }
#         , device:                   (String) {"cpu", "cuda"} device to
#                                     operate whenever possible
#         , epsilon:                  (float) probability of selecting a
#                                     random action
#         , beta:                     (float) parameter to determin the
#                                     level of compensation for bias when
#                                     computing priorities
#         , con_learner:              (multiprocessing.Connection) connection
#                                     where new weights are received and termination
#         , transition_queue:         (multiprocessing.Queue) SimpleQueue where
#                                     transitions are sent to replay buffer
#     }

#     """
     
#     # queues
#     con_learner = args["con_learner"]
#     transition_queue_to_memory = args["transition_queue_to_memory"] 
            
#     device = args["device"]

#     # local buffer of fixed size to store transitions before sending
#     local_buffer = [None] * args["size_local_memory_buffer"]
#     q_values_buffer = [None] * args["size_local_memory_buffer"]

#     # set network to eval mode
#     NN = args["model"]
#     if NN == NN_11 or NN == NN_17:
#         NN_config = args["model_config"]
#         model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
#     else:
#         model = NN()
    
#     model.to(device)
#     model.eval()
    
#     # env and env params
#     env = gym.make(args["env"], config=args["env_config"])

#     no_actions = int(env.action_space.high[-1])
#     grid_shift = int(env.system_size/2)
    
#     # Get initial network params
#     weights = None
#     while weights == None:
#         msg, weights = con_learner.recv() # blocking op
#         if msg != "weights":
#             weights = None

#     # load weights
#     vector_to_parameters(weights, model.parameters())


#     # init counters
#     steps_counter = 0
#     update_counter = 1
#     local_memory_index = 0
    
#     # startup
#     state = env.reset()
#     steps_per_episode = 0
#     terminal_state = False
   
#     # main loop over training steps
#     while True:
#         # print(steps_counter)
#         steps_counter += 1
#         if con_learner.poll():
#             msg, weights = con_learner.recv()
            
#             if msg == "weights":
#                 vector_to_parameters(weights, model.parameters())
            
#             elif msg == "prep_terminate":
#                 con_learner.send("ok")
#                 break

#         steps_per_episode += 1
#         previous_state = state
        
#         # select action using epsilon greedy policy
#         action, q_values = selectAction(number_of_actions=no_actions,
#                                         epsilon=args["epsilon"], 
#                                         grid_shift=grid_shift,
#                                         toric_size = env.system_size,
#                                         state = state,
#                                         model = model,
#                                         device = device)


#         state, reward, terminal_state, _ = env.step(action)

#         # generate transition to store in local memory buffer
#         transition = generateTransition(action,
#                                         reward,
#                                         grid_shift,
#                                         previous_state,
#                                         state,
#                                         terminal_state)

#         local_buffer[local_memory_index] = transition
#         q_values_buffer[local_memory_index] = q_values
#         local_memory_index += 1

#         if (local_memory_index >= (args["size_local_memory_buffer"])): 

#             priorities = computePriorities(local_buffer, q_values_buffer, args["discount_factor"])      
#             to_send = [*zip(local_buffer, priorities)]

#             # send buffer to learner
#             transition_queue_to_memory.put(to_send)
#             local_memory_index = 0

#         if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
#             state = env.reset()
#             steps_per_episode = 0
#             terminal_state = False
    
#     # ready to terminate
#     while True:
#         msg, _ = con_learner.recv()
#         if msg == "terminate":
#             transition_queue_to_memory.close()
#             con_learner.send("ok")
#             break


# def computePrioritiesOld(local_buffer, q_values_buffer, discount_factor):
#     """ Computes the absolute temporal difference value.

#     Parameters
#     ==========
#     local_buffer:        (list) local transitions from the actor
#     q_values_buffer:     (list) q values for respective action
#     discount_factor:     (float) discount future rewards

#     Returns
#     =======
#     (np.array) absolute TD error. (r + Qmax(st+1, a) - Q(st,a))
#     """

#     transitions             = Transition(*zip(*local_buffer))
#     reward_batch            = np.array(transitions.reward)
#     q_values_max_next_state = np.amax(np.roll(q_values_buffer, -1), axis=1)
#     q_values_state          = np.array([q_values_buffer[i][a.action-1] for i, a in enumerate(transitions.action)])

#     return np.absolute(reward_batch + discount_factor*q_values_max_next_state - q_values_state)

