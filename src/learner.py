# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter
from torch import from_numpy
# other
import numpy as np
import gym
from queue import Empty
# from file
from src.util import Action, Perspective, Transition, generatePerspective
from src.evaluation import evaluate

# debuging
import time


def learner(rank, world_size, args):
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
            , min_qbit_errors   (int)
            , p_error           (float)
        }
    }
    """

    def terminate():
        
        # prepare replay memory for termination
        msg = "prep_terminate"
        con_replay_memory.send(msg)
        #wait for acknowlage
        back = con_replay_memory.recv()    
        
        # prepare actors for termination
        msg = ("prep_terminate", None)
        for a in range(world_size-2):
            con_actors[a].send(msg)
            # wait for acknowledge
            back = con_actors[a].recv()
        
        # terminate actors
        msg = ("terminate", None)
        for a in range(world_size-2):
            con_actors[a].send(msg)
            # wait for acknowledge
            back = con_actors[a].recv()

        # empty and close queue before termination
        try:
            while True:
                transition_queue_from_memory.get_nowait()
        except Empty:
            pass
        
        transition_queue_from_memory.close()
        update_priorities_queue_to_memory.close()

        
        # terminate memory
        msg = "terminate"
        con_replay_memory.send(msg)
        # wait for acknowlage
        back = con_replay_memory.recv()
    
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
            # from np to tensor
            tensor = from_numpy(batch_input)
            tensor = tensor.type('torch.Tensor')
            return tensor.to(device)

        # get transitions and unpack them to minibatch
        batch = data[0]
        weights = data[1]
        index = data[2]

        weights = from_numpy(np.array(weights)).type('torch.Tensor').to(device)

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

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal, weights, index


    update_priorities_queue_to_memory = args["update_priorities_queue_to_memory"]
    transition_queue_from_memory = args["transition_queue_from_memory"]
    device = args["device"]
    train_steps = args["train_steps"]
    discount_factor = args["discount_factor"]
    batch_size = args["batch_size"]

    con_actors = args["con_actors"]
    con_replay_memory = args["con_replay_memory"]
    
    eval_freq = args["eval_freq"]
    env_config = args["env_config"]
    env = gym.make(args['env'], config=env_config)
    system_size = env_config["size"]
    grid_shift = int(system_size/2)


    # Init nets
    policy_class = args["policy_net"]
    policy_config = args["policy_config"] 
    policy_net = policy_class(policy_config["system_size"], policy_config["number_of_actions"], args["device"])
    
    target_class = args["target_net"]
    target_config = args["target_config"] 
    target_net = target_class(target_config["system_size"], target_config["number_of_actions"], args["device"])
    
    # Tensorboard
    tb = SummaryWriter(log_dir=args["tb_log_dir"]+"_learner", filename_suffix="_learner")
    update_tb = args["update_tb"]

    # init counter
    push_new_weights = 0

    # logging
    wait_time = 0
    sum_loss = 0
    sum_wait_time = 0


    # define criterion and optimizer
    criterion = nn.MSELoss(reduction='none')
    if args["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args["learning_rate"])
    elif args["optimizer"] == 'Adam':    
        optimizer = optim.Adam(policy_net.parameters(), lr=args["learning_rate"])
    

    # Push initial network params
    params = parameters_to_vector(policy_net.parameters()) 
    # weights = policy_net.state_dict()
    for actor in range(world_size-2):
        msg = ("weights", params.detach())
        con_actors[actor].send(msg)
    
    # Wait until replay memory has enough transitions for one batch
    while transition_queue_from_memory.empty(): continue

    # Start training
    for t in range(train_steps):
        print("learner: traning step: ",t+1," / ",train_steps)

        # wait until there is an item in queue
        while transition_queue_from_memory.empty():
            wait_time += 1
            continue 

        data = transition_queue_from_memory.get()

        # TODO: Everything out from here should be tenors, except indices
        batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal, weights, indices = dataToBatch(data)
        
        policy_net.train()
        target_net.eval()

        # compute policy net output
        policy_output = policy_net(batch_state)
        policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

        # compute target network output
        target_output = predictMax(target_net, batch_next_state, batch_size, grid_shift, system_size, device)
        target_output = target_output.to(device)

        # compute loss and update replay memory
        y = batch_reward + ((~batch_terminal) * discount_factor * target_output)
        loss = criterion(y, policy_output)
        
        # Compute priorities
        priorities = weights * loss.cpu()
        priorities = np.absolute(priorities.detach().numpy())
        
        optimizer.zero_grad()
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()

        # update priorities in replay buffer
        update_priorities_queue_to_memory.put([*zip(priorities, indices)])

        # update actor weights
        push_new_weights += 1
        if push_new_weights >= args["policy_update"]:
            params = parameters_to_vector(policy_net.parameters())
            msg = ("weights", params.detach())
            for actor in range(world_size-2):
                con_actors[actor].send(msg)
            push_new_weights = 0

        sum_loss += loss.sum()
        sum_wait_time += wait_time
        wait_time = 0

        # eval and write to tensorboard
        # if t % eval_freq == 0:
            # suc_rate, gr_st, avg_no_steps, mean_q, fail_syndr, p_errs = evaluate(policy_net
            #                                                                     , env
            #                                                                     , grid_shift
            #                                                                     , device
            #                                                                     , [0.1, 0.5]                  
            #                                                                     , num_of_predictions=10)

        # write to tensorboard        
        if t % update_tb == 0:
            tb.add_scalar('Avg Policy Loss', sum_loss.item()/eval_freq, t)
            tb.add_scalar('Avg Wait Time Learner For New Transitions', sum_wait_time/eval_freq, t)
            sum_loss = 0
            sum_wait_time = 0
        

    # training done
    torch.save(policy_net.state_dict(), "network/Size_{}_{}.pt".format(system_size, type(policy_net).__name__))
    tb.close()
    terminate()
    


def predictMax(model, batch_state, batch_size, grid_shift, system_size, device):
    """ Generates the max Q values for a batch of states.
    Params
    ======
    action_index: If the q value of the performed action is requested, 
    provide the chosen action index

    Return
    ======
    (torch.Tensor) A tensor containing the max q value for each state.
    """
    
    model.eval()

    # Create containers
    batch_output = np.zeros(batch_size)
    batch_perspectives = np.zeros(shape=(batch_size, 2, system_size, system_size))
    batch_actions = np.zeros(batch_size)
    for i in range(batch_size):
        if (batch_state[i].cpu().sum().item() == 0):
            batch_perspectives[i,:,:,:] = np.zeros(shape=(2, system_size, system_size))
        else:
            # Generate perspectives
            perspectives = generatePerspective(grid_shift, system_size, np.array(batch_state[i])) 
            perspectives = Perspective(*zip(*perspectives))
            perspectives = np.array(perspectives.perspective)
            perspectives = from_numpy(perspectives).type('torch.Tensor').to(device)

            # prediction
            with torch.no_grad():
                output = model(perspectives)
                q_values = np.array(output.cpu())
                row, col = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape) 
                batch_output[i] = q_values[row, col]      


    batch_output = from_numpy(batch_output).type('torch.Tensor')
    return batch_output
