import torch
import torch.distributed as dist
import numpy as np 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import namedtuple
from torch import from_numpy
import random


Perspective = namedtuple('Perspective', ['perspective', 'position'])
Action = namedtuple('Action', ['position', 'action'])
Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])

def actor(rank, world_size, weight_queue, transition_queue, args):
    
    print("actor_",rank,": Hello") 
    """ 
    args = {"train_steps", 
            "max_actions_per_episode", 
            "update_policy",
            "size_local_memory_buffer", 
            "min_qubit_errors", 
            "model",
            "env",
            "device",
            "epsilon"
            }
    """
     
    device = args["device"]
    
    # set network to eval mode
    model = args["model"]
    model.eval()
    
    env = args["env"]

    no_actions = int(env.action_space.high[-1])
    grid_shift = int(env.system_size/2)
    
    # Get initial network params
    weights = None
    while weights == None:
        if not weight_queue.empty():
            weights = weight_queue.get()[0]

    # load weights
    vector_to_parameters(weights, model.parameters())
    
    # init counters
    steps_counter = 0
    update_counter = 1
    # iteration = 0

    # local buffer of fixed size to store transitions before sending
    local_buffer = [None] * args["size_local_memory_buffer"]   
    local_memory_index = 0
    
    state = env.reset()
    steps_per_episode = 0
    terminal_state = False
   
         
    print("actor_",rank,": Starting exploration") 
    # main loop over training steps 
    for iteration in range(args["train_steps"]):
        
        print("actor_",rank,": iteration: ",iteration) 
        #steps_counter += 1 # Just use iteration
        steps_per_episode += 1
        previous_state = state
        
        # select action using epsilon greedy policy
        print("actor_",rank,": call select action") 
        action, q_value = select_action(number_of_actions=no_actions,
                                    epsilon=args["epsilon"], 
                                    grid_shift=grid_shift,
                                    toric_size = env.system_size,
                                    state = state,
                                    model = model,
                                    device = device,
                                    env = env)
        
        print("actor_",rank,": Done selecting action") 
        state, reward, terminal_state, _ = env.step(action)
        print("done with step")
        print("actor_",rank,": step with action: ",action) 

        # generate transition to stor in local memory buffer
        transition = generateTransition(action,
                                            reward,
                                            grid_shift,
                                            previous_state,
                                            state,
                                            terminal_state)

        # TODO: Store q_value
        local_buffer.insert(local_memory_index, transition)
        local_memory_index += 1

        if (local_memory_index % len(local_buffer)):
            # TODO: Compute priorities
                # initial priority

                # weights
                # normalize weights
                # priority = weights * initial priority
            # TODO (Adam) send buffer to learner
            local_memory_index = 0

        # if new weights are available, update network
        #if not weight_queue.empty():
        #    w = weight_queue.get()[0]
        #    vector_to_parameters(w, model.parameters())
        
        if terminal_state or steps_per_episode > args["max_actions_per_episode"]:
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False
            

def select_action(number_of_actions, epsilon, grid_shift,
                    toric_size, state, model, device, env):
    # set network in evluation mode 
    model.eval()
    # generate perspectives 
    perspectives = generatePerspective(grid_shift, toric_size, state)
    number_of_perspectives = len(perspectives)
    # preprocess batch of perspectives and actions

    print(perspectives) 
    perspectives = Perspective(*zip(*perspectives))
    batch_perspectives = np.array(perspectives.perspective)
    batch_perspectives = from_numpy(batch_perspectives).type('torch.Tensor')    
    batch_perspectives = batch_perspectives.to(device)
    batch_position_actions = perspectives.position
    
    print("actor_: done batch perspectives")

    for p in model.parameters():
        print(p)

    # Policy
    policy_net_output = None
    q_values_table = None
    with torch.no_grad():
        policy_net_output = model(batch_perspectives)
        q_values_table = np.array(policy_net_output.cpu())

    #choose action using epsilon greedy approach
    rand = random.random()
    if(1 - epsilon > rand):
        print("actor_: select greedy") 
        # select greedy action 
        print("actor_: select greedy: output from model") 
        row, col = np.where(q_values_table == np.max(q_values_table))
        perspective = row[0]
        max_q_action = col[0] + 1
        action = [  batch_position_actions[perspective][0],
                    batch_position_actions[perspective][1],
                    batch_position_actions[perspective][2],
                    max_q_action]
        q_value = q_values_table[row, col]

    # select random action
    else:
        print("actor_: select random")
        random_perspective = random.randint(0, number_of_perspectives-1)
        random_action = random.randint(1, number_of_actions)
        action = [  batch_position_actions[random_perspective][0],
                    batch_position_actions[random_perspective][1],
                    batch_position_actions[random_perspective][2],
                    random_action]
        q_value = [random_perspective, random_action]

    return action, q_value
    

def generatePerspective(grid_shift, toric_size, state):
    def mod(index, shift):
        index = (index + shift) % toric_size 
        return index
    perspectives = []
    vertex_matrix = state[0,:,:]
    plaquette_matrix = state[1,:,:]
    # qubit matrix 0
    for i in range(toric_size):
        for j in range(toric_size):
            if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
            plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                temp = Perspective(new_state, (0,i,j))
                perspectives.append(temp)
    # qubit matrix 1
    for i in range(toric_size):
        for j in range(toric_size):
            if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
            plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                new_state = rotate_state(new_state) # rotate perspective clock wise
                temp = Perspective(new_state, (1,i,j))
                perspectives.append(temp)
    
    return perspectives

def generateTransition( action, 
                        reward, 
                        grid_shift,       
                        previous_state,   #   Previous state before action
                        state,            #   Current state    
                        terminal_state    #   True/False
                        ):
    

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

def rotate_state(state):
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        rot_plaquette_matrix = np.rot90(plaquette_matrix)
        rot_vertex_matrix = np.rot90(vertex_matrix)
        rot_vertex_matrix = np.roll(rot_vertex_matrix, 1, axis=0)
        rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
        return rot_state 

    
def shift_state(row, col, previous_state, state, grid_shift):
        previous_perspective = np.roll(previous_state, grid_shift-row, axis=1)
        previous_perspective = np.roll(previous_perspective, grid_shift-col, axis=2)
        perspective = np.roll(state, grid_shift-row, axis=1)
        perspective = np.roll(perspective, grid_shift-col, axis=2)
        return previous_perspective, perspective
