
#
#    """
#        Learner Process
#    """
#    learner_args = {
#        "train_steps"                   :learner_training_steps,
#        "batch_size"                    :batch_size,
#        "learning_rate"                 :learner_learning_rate,
#        "policy_update"                 :learner_policy_update,
#        "discount_factor"               :transition_priorities_discount_factor,
#        "optimizer"                     :learner_optimizer,
#        "model"                         :model,
#        "model_config"                  :model_config,
#        "device"                        :learner_device,
#        "eval_freq"                     :learner_eval_freq,
#        "env"                           :env,
#        "env_config"                    :env_config,
#        "mpi_base_comm"                 :base_comm,
#        "mpi_comm_actor_learner"        :comm_actor_learner,
#        "mpi_base_comm_setup"           :base_comm_setup
#    }


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

def learner(args, memory_args):
    
    training_steps = args["train_steps"]
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

    #Brodcast weights
    msg = ("weights", params.detach())
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

    # Wait untill replay memory has enough transitions
    #TODO
    
     
    # Start training
    for t in range(train_steps):
        
        # Gather transitions from actors
        # TODO: fix format of what is sent from actor
        if t % gather_new_weights == 0:
           actor_transitions = base_comm.gather(actor_transitions, root = learner_rank)
           for i in range(len(actor_transitions)):
                replay_memory.save(actor_transitions[i].transition, actor_transitions[i])  
        
        print("learner: traning step: ",t+1," / ",train_steps)
        
        transitions, wights, indices = replay_memory.sample(batch_size, memory_beta)
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

        # update priorities in replay buffer
        #TODO
        update_priorities_queue_to_memory.put([*zip(priorities, indices)])

        # update actor weights
        #TODO
        push_new_weights += 1
        if push_new_weights >= args["policy_update"]:
            params = parameters_to_vector(policy_net.parameters())
            # update policy network
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device) # dont know if this is needed
            msg = ("weights", params.detach())
            # send weights to actors
            for actor in range(world_size-2):
                con_actors[actor].send(msg)
            push_new_weights = 0



    # training done
    torch.save(policy_net.state_dict(), "network/Size_{}_{}.pt".format(system_size, type(policy_net).__name__))
