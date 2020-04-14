import torch
import numpy as np 
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy
from src.nn.torch.NN import NN_11, NN_17
from src.evaluation import evaluate
import time

could_import_tb=True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    could_import_tb=False
    print("Could not import tensorboard. No logging will occur.")

import gym


def evaluator(args):
    

    
    NN              = args["model"]
    model_no_params = args["model_no_params"]
    if NN == NN_11 or NN == NN_17:
        NN_config = args["model_config"]
        model = NN(NN_config["system_size"], NN_config["number_of_actions"], args["device"])
    else:
        model = NN()

    device = args["device"]
    
    # eval params
    eval_p_errors       = args["eval_p_errors"]
    eval_no_episodes    = args["eval_no_episodes"]
    eval_freq           = args["eval_freq"]
    count_to_eval       = 0

    save_date           = args["save_date"]
    env_config      = args["env_config"]
    system_size     = env_config["size"]
    
    shared_mem_weights     = args["shared_mem_weights"]
    shared_mem_weight_id   = args["shared_mem_weight_id"]
    current_weight_id      = 0
    policy_update          = args["policy_update"]
    
    print("evaluator started on device: {}".format(device))

    if eval_freq != -1 and could_import_tb:
        tb = SummaryWriter("runs/{}/Evaluator/".format(save_date))

        # load initial network weights
        weights = np.empty(model_no_params)
        with shared_mem_weights.get_lock():
            reader = np.frombuffer(shared_mem_weights.get_obj())
            np.copyto(weights, reader)
        vector_to_parameters(from_numpy(weights).to(device).type(torch.FloatTensor), model.parameters())
        
        model.to(device)
        model.eval()
        
        new_weights = False

        while True:
            
            
            with shared_mem_weights.get_lock():
                if (current_weight_id + eval_freq) < shared_mem_weight_id.value:
                    reader = np.frombuffer(shared_mem_weights.get_obj())
                    np.copyto(weights, reader)
                    current_weight_id = shared_mem_weight_id.value
                    new_weights = True
             
            if new_weights == False:
                time.sleep(1)
                    
            
            # Evaluate network
            if new_weights:
                new_weights = False
                learning_step = current_weight_id * policy_update
                vector_to_parameters(from_numpy(weights).type(torch.FloatTensor).to(device), model.parameters())
    
                print("start evaluat of network_id: {}".format(current_weight_id))

                success_rate, ground_state_rate, _, mean_q_list, _ = evaluate(  model,
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
                    tb.add_scalar("Network/Mean Q, p error {}".format(p), mean_q_list[i], learning_step)
                    tb.add_scalar("Network/Success Rate, p error {}".format(p), success_rate[i], learning_step)
                    tb.add_scalar("Network/Ground State Rate, p error {}".format(p), ground_state_rate[i], learning_step)
