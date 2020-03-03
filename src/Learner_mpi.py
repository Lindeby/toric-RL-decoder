
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# other
import numpy as np
import gym

# from file
from src.util_learner import predictMaxOptimized, dataToBatch
from src.evaluation import evaluate
from src.ReplayMemory import PrioritizedReplayMemory

from src.nn.torch.NN import NN_11, NN_17

from copy import deepcopy

import time

def learner(args, memory_args):
    start_time = time.time() 
    train_steps = args["train_steps"]
    batch_size = args["batch_size"]
    learning_rate = args["learning_rate"]
    policy_update = args["policy_update"]
    discount_factor = args["discount_factor"]
    device = args["device"]
    eval_freq = args["eval_freq"]
    base_comm = args["mpi_base_comm"]
    learner_rank = args["mpi_learner_rank"]
    env_config = args["env_config"]
    system_size = env_config["size"]
    grid_shift = int(env_config["size"]/2)
    synchronize = args["synchronize"]

    world_size = base_comm.Get_size()


    #Memory
    memory_capacity = memory_args["capacity"]
    memory_alpha = memory_args["alpha"]
    memory_beta = memory_args["beta"]
    replay_size_before_sampling = memory_args["replay_size_before_sampling"]
    
    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)
    

    # Init policy net
    policy_class = args["model"]
    policy_config = args["model_config"] 
    if policy_class == NN_11 or policy_class == NN_17:
        policy_net = policy_class(policy_config["system_size"], policy_config["number_of_actions"], args["device"])
        target_net = policy_class(policy_config["system_size"], policy_config["number_of_actions"], args["device"]) 
    else:
        policy_net = policy_class()
        target_net = policy_class()

    policy_net.to(device)
    target_net.to(device)
    
    # copy policy params to target
    params = parameters_to_vector(policy_net.parameters()) 
    vector_to_parameters(params, target_net.parameters())
    w = params.detach().to('cpu')
    msg = ("weights", w)
    base_comm.bcast(msg, root=learner_rank)
    
    # define criterion and optimizer
    optimizer = None
    criterion = nn.MSELoss(reduction='none')
    if args["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args["learning_rate"])
    elif args["optimizer"] == 'Adam':    
        optimizer = optim.Adam(policy_net.parameters(), lr=args["learning_rate"])

    # init counter
    push_new_weights = 0


    
    while replay_memory.filled_size() < replay_size_before_sampling:

        params = parameters_to_vector(policy_net.parameters())
        # update policy network
        vector_to_parameters(params, target_net.parameters())
        target_net.to(device) # dont know if this is needed
        # broadcast weights
        w = params.detach().to('cpu')
        msg = ("weights", w)
        base_comm.bcast(msg, root=learner_rank)
        # gather transitions from actors
        actor_transitions = []
        actor_transitions = base_comm.gather(actor_transitions, root = learner_rank)
        # save transitions in replay memory
        for a in range(0, world_size):
            if a == learner_rank:
                continue
            a_transitions = actor_transitions[a]
            
            for i in range(len(a_transitions)):
                replay_memory.save(a_transitions[i][0], a_transitions[i][1])
         
    t = None
    # Start training
    for t in range(train_steps):
        
        # Time guard
        if time.time() - start_time > args["job_max_time"]:
            print("time exeded")
            break
        
        # Synchronisation 
        # - Gather transitions from actors
        # - Update policy network
        # - Send new weights to actors
        if t % synchronize == 0:

            params = parameters_to_vector(policy_net.parameters())
            # update policy network
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device) # dont know if this is needed
            # broadcast weights
            w = params.detach().to('cpu')
            msg = ("weights", w)
            base_comm.bcast(msg, root=learner_rank)
            # gather transitions from actors
            actor_transitions = []
            actor_transitions = base_comm.gather(actor_transitions, root = learner_rank)
            # save transitions in replay memory
            for a in range(0, world_size):
                if a == learner_rank:
                    continue
                a_transitions = actor_transitions[a]
                
                for i in range(len(a_transitions)):
                    replay_memory.save(a_transitions[i][0], a_transitions[i][1])

        #print("learner: traning step: ",t+1," / ",train_steps)
        
        transitions, weights, indices = replay_memory.sample(batch_size, memory_beta)
        data = (transitions, weights, indices)

        batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal, weights, indices = dataToBatch(data, device)
        
        policy_net.train()
        target_net.eval()

        # compute policy net output
        policy_output = policy_net(batch_state)
        policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

        # compute target network output
        # target_output = predictMax(target_net, batch_next_state, len(batch_next_state),grid_shift, system_size, device)
        target_output = predictMaxOptimized(target_net, batch_next_state, grid_shift, system_size, device)
        
        target_output = target_output.to(device)

        # compute loss and update replay memory
        y = batch_reward + ((~batch_terminal).type(torch.float) * discount_factor * target_output)
        y = y.clamp(-100, 100)
        loss = criterion(y, policy_output)
        loss = weights * loss
        
        # Compute priorities
        priorities = np.absolute(loss.cpu().detach().numpy())
        
        optimizer.zero_grad()
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()

        # update priorities in replay_memory
        replay_memory.priority_update(indices, priorities)
    
    # training done
    msg = ("terminate", None)
    base_comm.bcast(msg, root=learner_rank)
    torch.save(policy_net.state_dict(), "network/mpi/Size_{}_{}.pt".format(system_size, type(policy_net).__name__))

    stop_time = time.time()
    elapsed_time = stop_time - start_time 
    print("elapsed time: ",elapsed_time)
    print("learning steps: ",t)
