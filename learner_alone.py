
import torch
import torch.nn as nn
import torch.optim as optim


from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy
# other
import numpy as np
#import gym
#import gym_ToricCode
# from file
from src.util import Action, Perspective, Transition, generatePerspective
from src.evaluation import evaluate
from src.nn.torch.NN import NN_17
# debuging
#import time

from numpy import load

from src.learner import predictMax, predictMaxOptimized

def learner():
    """The learner in a distributed RL setting. Updates the network params, pushes
    new network params to actors. Additionally, this function collects the transitions
    in the queue from the actors and manages the replay buffer.

    Params
    ======
    rank:           (int)
    world_size:     (int)
    args: (dict) 
    {
        no_actors:                            (int)
        , train_steps:                        (int)
        , batch_size:                         (int)
        , optimizer:                          (String)
        , policy_net:                         (torch.nn)
        , policy_config:                      (dict)
        {
            system_size:        (int) size of the toric grid.
            , number_of_actions (int)
        }
        , target_net:                         (torch.nn)
        , target_config:                      (dict)
        {
            system_size:        (int) size of the toric grid.
            , number_of_actions (int)
        }
        , learning_rate:                      (float)
        , device:                             (String) {"cpu", "cuda"}
        , policy_update:                      (int)
        , discount_factor:                    (float)
        , con_send_weights:                   (multiprocessing.Connection)
        , transition_queue_from_memory:       (multiprocessing.Queue) Queue
        , update_priorities_queue_to_memory:  (multiprocessing.Queue) Queue
        , con_actors:                         Array of connections (multiprocessing.Pipe)  Pipe(Duplex = True)
        , con_replay_memory:                  (multiprocessing.Pipe)  Pipe(Duplex = True)
        , eval_freq                           (int)
        , update_tb                           (int) frequensy to update tensorboard
        , tb_log_dir                          (String) tensorboard log dir
        , env:                                (String) for evaluating the policy.
        , env_config:                         (dict)
        
            size:               (int)
            , min_qubit_errors   (int)
            , p_error           (float)
        }
    }
    """

    
    def dataToBatch(data):
        """ Converts data from the replay memory to appropriate dimensions.

        Params
        ======
        data: () Data from the replay buffer queue. Each item is a tuple of
                 (('state', 'action', 'reward', 'next_state', 'terminal'), weight, index)
        Returns
        =======
        """

        def toNetInput(batch):
            batch_input = np.stack(batch, axis=0) 
            tensor = from_numpy(batch_input)
            tensor = tensor.type('torch.Tensor')
            return tensor.to(device)

        # get transitions and unpack them to minibatch
        batch = data
        #weights = data[1]
        #index = data[2]

        #weights = from_numpy(np.array(weights)).type('torch.Tensor').to(device)

        # preprocess batch_input and batch_target_input for the network
        list_state, list_action, list_reward, list_next_state, list_terminal = zip(*batch)

        batch_state = toNetInput(list_state)
        batch_next_state = toNetInput(list_next_state)

        # unpack action batch
        batch_action = Action(*zip(*list_action))
        batch_actions = np.array(batch_action.action) - 1 # -1 to compensate for env action index
        batch_actions = torch.Tensor(batch_actions).long()
        batch_actions = batch_actions.to(device)

        # preprocess batch_terminal and batch reward
        batch_terminal = from_numpy(np.array(list_terminal)).to(device)
        batch_reward   = from_numpy(np.array(list_reward)).type('torch.Tensor').to(device)

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal


    device = 'cpu'
    train_steps = 64 #TODO: untill no more transitions?
    discount_factor = 0.9
    batch_size = 32 #TODO: not sure yet

    
    # eval params
    system_size = 9
    grid_shift = int(system_size/2)
 
    #env_config = {  "size": system_size,
    #                "min_qubit_errors": 0,
    #                "p_error": 0.1
    #             }
    #env = gym.make("toric-code-v0", config = env_config)
    
    
    # Init policy net
    policy_net = NN_17(system_size, 3, device)
    policy_net.to(device)

    target_net = NN_17(system_size, 3, device)
    target_net.to(device)

    # define criterion and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)

    memory = load('transitions_2048.npy', allow_pickle=True)
    def get_batch_from_mem(iteration, b_size):

        start = iteration*b_size
        return memory[start: start+b_size] 
    
    # Start training
    for t in range(train_steps):
        print("learner: traning step: ",t+1," / ",train_steps)

        data = get_batch_from_mem(t, batch_size)

        batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal = dataToBatch(data)
        
        policy_net.train()
        target_net.eval()

        # compute policy net output
        policy_output = policy_net(batch_state)
        policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

        
        # compute target network output
        target_output = predictMaxOptimized(target_net, batch_next_state, grid_shift, system_size, device)
        #target_output = predictMax(target_net, batch_next_state, batch_size, grid_shift, system_size, device)
        target_output = target_output.to(device)

        # compute loss and update replay memory
        y = batch_reward + ((~batch_terminal) * discount_factor * target_output)
        loss = criterion(y, policy_output)
        
        # Compute priorities
        #priorities = weights * loss.cpu()
        #priorities = np.absolute(priorities.detach().numpy())
        
        optimizer.zero_grad()
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()



    ## training done
    ##torch.save(policy_net.state_dict(), "network/Size_{}_{}.pt".format(system_size, type(policy_net).__name__))
    
learner()

