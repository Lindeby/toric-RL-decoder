# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

# other
import numpy as np
import gym
from queue import Empty

# from file
from src.util_learner import predictMaxOptimized, dataToBatch
from src.evaluation import evaluate

# Quality of life
from src.nn.torch.NN import NN_11, NN_17


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

    # Tensorboard
    tb = SummaryWriter(log_dir=args["tb_log_dir"]+"_learner", filename_suffix="_learner")
    update_tb = args["update_tb"]

    update_priorities_queue_to_memory = args["update_priorities_queue_to_memory"]
    transition_queue_from_memory = args["transition_queue_from_memory"]
    device = args["device"]
    train_steps = args["train_steps"]
    discount_factor = args["discount_factor"]
    batch_size = args["batch_size"]

    con_actors = args["con_actors"]
    con_replay_memory = args["con_replay_memory"]
    
    # eval params
    eval_freq = args["eval_freq"]
    env_config = args["env_config"]
    system_size = env_config["size"]
    grid_shift = int(env_config["size"]/2)

    # Init policy net
    policy_class = args["policy_net"]
    policy_config = args["policy_config"] 
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

    # Push initial network params
    for actor in range(world_size-2):
        msg = ("weights", params.detach())
        con_actors[actor].send(msg)

    # define criterion and optimizer
    criterion = nn.MSELoss(reduction='none')
    if args["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args["learning_rate"])
    elif args["optimizer"] == 'Adam':    
        optimizer = optim.Adam(policy_net.parameters(), lr=args["learning_rate"])


    # init counter
    push_new_weights = 0

    # logging
    wait_time = 0
    sum_loss = 0
    sum_wait_time = 0
    
    print("Learner waiting for replay memory to be filled.")
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
        
        # Compute priorities
        priorities = weights * loss
        priorities = np.absolute(priorities.cpu().detach().numpy())
        
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
            # update policy network
            vector_to_parameters(params, target_net.parameters())
            target_net.to(device) # dont know if this is needed
            msg = ("weights", params.detach())
            # send weights to actors
            for actor in range(world_size-2):
                con_actors[actor].send(msg)
            push_new_weights = 0

        sum_loss += loss.sum()
        sum_wait_time += wait_time
        wait_time = 0


        # eval and write to tensorboard
        if t % eval_freq == 0 :
            p_errors = [0.1]
            suc_rate, gr_st, avg_no_steps, mean_q, _ = evaluate(policy_net
                                                                , args["env"]
                                                                , args["env_config"]
                                                                , grid_shift
                                                                , device
                                                                , p_errors                  
                                                                , num_of_episodes=1)
            for i,e in enumerate(p_errors):
                tb.add_scalar("Eval/SuccessRate_{}".format(e), suc_rate[i], t)
                tb.add_scalar("Eval/GroundState_{}".format(e), gr_st[i], t)
                tb.add_scalar("Eval/AvgNoSteps_{}" .format(e), avg_no_steps[i], t)
                tb.add_scalar("Eval/MeanQValue_{}" .format(e), mean_q[i], t)

        # write to tensorboard        
        if t % update_tb == 0:
            tb.add_scalar('Eval/Avg_Over_{}_Loss'.format(update_tb), sum_loss.item()/eval_freq, t)
            tb.add_scalar('Wait/Avg_Over_{}_Wait_Learner_For_New_Transitions'.format(update_tb), sum_wait_time/eval_freq, t)
            sum_loss = 0
            sum_wait_time = 0
        

    # training done
    torch.save(policy_net.state_dict(), "network/Size_{}_{}.pt".format(system_size, type(policy_net).__name__))
    tb.close()
    terminate()
