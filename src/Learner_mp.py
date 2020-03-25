
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy
could_import_tb=True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    could_import_tb=False
    print("Could not import tensorboard. No logging will occur.")

# other
import numpy as np
import gym

# from file
from src.util_learner import predictMaxOptimized, dataToBatch
from src.evaluation import evaluate

# Quality of life
from src.nn.torch.NN import NN_11, NN_17

import time


def learner(args):
    start_time = time.time() 

    train_steps     = args["train_steps"]
    discount_factor = args["discount_factor"]
    batch_size      = args["batch_size"]
    device          = args["device"]

    # params
    env_config      = args["env_config"]
    system_size     = env_config["size"]
    grid_shift      = int(env_config["size"]/2)
    policy_update   = args["policy_update"]
    save_date       = args["save_date"]
    actor_env_p_error_strategy = args["actor_env_p_error_strategy"]

    # eval params
    eval_p_errors       = args["learner_eval_p_errors"]
    eval_no_episodes    = args["learner_eval_no_episodes"]
    eval_freq           = args["learner_eval_freq"]
    count_to_eval       = 0
    if eval_freq != -1 and could_import_tb:
        tb = SummaryWriter("runs/{}/Learner/".format(save_date))

    # Comms
    learner_io_queue        = args["learner_io_queue"]
    io_learner_queue        = args["io_learner_queue"]
    shared_mem_weights      = args["shared_mem_weights"]
    shared_mem_weight_id    = args["shared_mem_weight_id"]

    
    # Init networks
    policy_class    = args["model"]
    policy_config   = args["model_config"] 
    model_no_params = args["model_no_params"]
    if policy_class == NN_11 or policy_class == NN_17:
        policy_net = policy_class(policy_config["system_size"], policy_config["number_of_actions"], device)
        target_net = policy_class(policy_config["system_size"], policy_config["number_of_actions"], device) 
    else:
        policy_net = policy_class()
        target_net = policy_class()
    
    # Load initial parameters
    weights = np.empty(model_no_params)
    with shared_mem_weights.get_lock():
        reader = np.frombuffer(shared_mem_weights.get_obj())
        np.copyto(weights, reader)
    vector_to_parameters(from_numpy(weights).type(torch.FloatTensor), policy_net.parameters())
    vector_to_parameters(from_numpy(weights).type(torch.FloatTensor), target_net.parameters())
    policy_net.to(device)
    target_net.to(device)
    
    
    # define criterion and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = None
    if args["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args["learning_rate"])
    elif args["optimizer"] == 'Adam':    
        optimizer = optim.Adam(policy_net.parameters(), lr=args["learning_rate"])

    
    preformance_start = time.time()
    preformance_stop = None
    # Start training
    print("Learner: starting training loop.")
    for t in range(train_steps):
        #print("Learner timestep: {}".format(t))
        # Time guard
        if time.time() - start_time > args["job_max_time"]:
            print("Learner: time exceeded, aborting...")
            break
        
        # update target and update shared memory with new weights
        if t % policy_update == 0 and t != 0:
            performence_stop = time.time()
            performence_elapsed = performence_stop - preformance_start
            performence_transitions = policy_update * batch_size
            #print("consuming ",performence_transitions/performence_elapsed, "tranistions/s")
            preformance_start = time.time()
            params = parameters_to_vector(policy_net.parameters()) # get policy weights
            vector_to_parameters(params, target_net.parameters())  # load policy weights to target
            target_net.to(device)

            # update shared memory with new weights
            with shared_mem_weights.get_lock():
                shared_mem_weights[:]        = params.detach().cpu().numpy()
                shared_mem_weight_id.value += 1

        if io_learner_queue.qsize == 0:
            print("Learner waiting")
        data = io_learner_queue.get()
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
        optimizer.zero_grad()

        loss = weights * loss
        
        # Compute priorities
        priorities = np.absolute(loss.cpu().detach().numpy())
        
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()

        # update priorities in replay_memory
        p_update = (indices, priorities)
        msg = ("priorities", p_update)
        learner_io_queue.put(msg)


        # evaluations of policy
        count_to_eval += 1
        if eval_freq != -1 and could_import_tb and count_to_eval >= eval_freq:
            count_to_eval = 0
            success_rate, ground_state_rate, _, mean_q_list, _ = evaluate(  policy_net,
                                                                            'toric-code-v0',
                                                                            env_config,
                                                                            int(system_size/2),
                                                                            device,
                                                                            eval_p_errors,
                                                                            num_of_episodes=eval_no_episodes,
                                                                            epsilon=0.0,
                                                                            num_of_steps=75,
                                                                            plot_one_episode=False, 
                                                                            minimum_nbr_of_qubit_errors=0)
            for i, p in enumerate(eval_p_errors):
                tb.add_scalar("Network/Mean Q, p error {}".format(p), mean_q_list[i], t)
                tb.add_scalar("Network/Success Rate, p error {}".format(p), success_rate[i], t)
                tb.add_scalar("Network/Ground State Rate, p error {}".format(p), ground_state_rate[i], t)

    # close tensorboard writer
    if eval_freq != -1 and could_import_tb:
        tb.close()

    # training done
    # save network
    msg = ("terminate", None)
    learner_io_queue.put(msg)
    save_path = "runs/{}/Size_{}_{}_{}_{}.pt".format(save_date, system_size, type(policy_net).__name__, actor_env_p_error_strategy, save_date)
    torch.save(policy_net.state_dict(), save_path)
    print("Saved network to {}".format(save_path))
    print("Total trainingsteps: {}".format(t))
     
