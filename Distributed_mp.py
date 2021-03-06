from src.nn.torch.ResNet import ResNet18
from src.nn.torch.NN import NN_11, NN_17
from src.Actor_mp import actor
from src.IO_mp import io
from src.Learner_mp import learner
from src.util_actor import calculateEpsilon

import numpy as np
import multiprocessing as mp
from multiprocessing.sharedctypes import Array, Value
from torch.nn.utils import parameters_to_vector
from datetime import datetime
import time, torch, sys

could_import_tb=True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    could_import_tb=False
    print("Could not import tensorboard. No logging will occur.")


def start_distributed_mp():

    # Setup

    # To continue training, give path to state dict
    state_dict_path = None 

    
    # Learner specific
    learner_training_steps   = 1000000
    learner_learning_rate    = 0.00025
    learner_policy_update    = 50
    learner_optimizer        = 'Adam'
    learner_device           = 'cuda'
    learner_job_max_time     = 60*60*24 -60*10 #2 hours 58min
    learner_save_date        = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    learner_eval_p_errors    = [0.1, 0.2, 0.3]
    learner_eval_no_episodes = 10
    learner_eval_freq        = -1 # -1 for no logging
   
    # Actor specific
    actor_max_actions_per_episode  = 75
    actor_size_local_memory_buffer = 100
    actor_no_envs       = 16           #number of envs/actor
    no_cuda_actors      = 1
    no_cpu_actors       = 0
    actor_no_actors     = no_cuda_actors + no_cpu_actors
    #epsilon             = calculateEpsilon(0.8, 7, actor_no_actors * actor_no_envs)
    epsilon             = calculateEpsilon(0.8, 7, actor_no_envs)
    epsilon_delta       = 0.005
    env_p_error_interval_start    = 0.1
    env_p_error_interval_final    = 0.3
    env_p_error_interval_increase = 0.00005
    env_p_error_strategy          = 'random' # either {'random', 'linear'}
    
    # Replay Memory specific
    replay_memory_size                  = 1000000
    replay_memory_alpha                 = 0.6
    replay_memory_beta                  = 0.4
    replay_memory_size_before_sampeling = 5000#replay_memory_size*0.05
    replay_memory_batch_in_queue_limit  = 10 #number of batches in queue to learner
    log_priority_dist                   = True
    log_write_frequency                 = 500
    log_priority_sample_max             = 10
    log_priority_sample_interval_size   = 0.01
    
    # Shared
    batch_size = 16
    discount_factor = 0.95
    env = "toric-code-v0"
    env_config = {  "size":9,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
            }

    model = ResNet18
    #model = NN_11
    #model = NN_17
    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }

    if not state_dict_path == None: 
        checkpoint = torch.load(state_dict_path, map_location=learner_device)
    else:
        checkpoint = None

    # Pre-load initial network weights
    if model == NN_11 or model == NN_17:
        m = model(model_config["system_size"], model_config["number_of_actions"], learner_device)
    else: 
        m = model()

    # load checkpoint params
    if not state_dict_path == None: 
        m.load_state_dict(checkpoint['model_state_dict'])

    params      = parameters_to_vector(m.parameters()).detach().cpu().numpy()
    no_params   = len(params)
    
    #Comm setup 
    actor_io_queue = mp.Queue()
    learner_io_queue = mp.Queue()
    io_learner_queue = mp.Queue()
    shared_mem_weight_id  = Value('i')
    shared_mem_weight_id.value = 0

    # Write initial weights to shared memory
    shared_mem_weights    = Array('d', no_params)            # Shared memory for weights
    mem_reader = np.frombuffer(shared_mem_weights.get_obj()) # create memory reader for shared mem
    np.copyto(mem_reader, params)                            # Write params to shared mem
    
    del m # delete tmp model to load network params to free up mem
    
    """
        Learner Process
    """
    learner_args = {
        "train_steps"                   :learner_training_steps,
        "batch_size"                    :batch_size,
        "learning_rate"                 :learner_learning_rate,
        "policy_update"                 :learner_policy_update,
        "discount_factor"               :discount_factor,
        "optimizer"                     :learner_optimizer,
        "model"                         :model,
        "model_config"                  :model_config,
        "model_no_params"               :no_params,
        "device"                        :learner_device,
        "env"                           :env,
        "env_config"                    :env_config,
        "job_max_time"                  :learner_job_max_time,
        "save_date"                     :learner_save_date,
        "learner_io_queue"              :learner_io_queue,
        "io_learner_queue"              :io_learner_queue,
        "shared_mem_weights"            :shared_mem_weights,
        "shared_mem_weight_id"          :shared_mem_weight_id,
        "learner_eval_p_errors"         :learner_eval_p_errors,
        "learner_eval_no_episodes"      :learner_eval_no_episodes,
        "learner_eval_freq"             :learner_eval_freq,
        "learner_checkpoint"            :checkpoint
    }
    
    
    """
        Memory Process
    """
    mem_args = {
        "capacity"                          :replay_memory_size,
        "alpha"                             :replay_memory_alpha,
        "beta"                              :replay_memory_beta,
        "batch_size"                        :batch_size,
        "io_learner_queue"                  :io_learner_queue,
        "learner_io_queue"                  :learner_io_queue,
        "actor_io_queue"                    :actor_io_queue,
        "batch_in_queue_limit"              :replay_memory_batch_in_queue_limit,
        "no_actors"                         :actor_no_actors,
        "replay_size_before_sampling"       :replay_memory_size_before_sampeling if not (replay_memory_size_before_sampeling is None) else min(batch_size, int(replay_memory_size*0.1)),
        "save_date"                         :learner_save_date,
        "log_priority_dist"                 :log_priority_dist,
        "log_write_frequency"               :log_write_frequency,
        "log_priority_sample_max"           :log_priority_sample_max,
        "log_priority_sample_interval_size" :log_priority_sample_interval_size,
        "start_time"                        :learner_save_date
    }
    
              
    """
        Actor Processes
    """
    actor_args = { 
        "max_actions_per_episode"       :actor_max_actions_per_episode, 
        "size_local_memory_buffer"      :actor_size_local_memory_buffer, 
        "env_config"                    :env_config,
        "model"                         :model,
        "model_config"                  :model_config,
        "model_no_params"               :no_params,
        "env"                           :env,
        "discount_factor"               :discount_factor,
        "no_envs"                       :actor_no_envs,
        "actor_io_queue"                :actor_io_queue,
        "shared_mem_weights"            :shared_mem_weights,
        "shared_mem_weight_id"          :shared_mem_weight_id,
        "epsilon_delta"                 :epsilon_delta,
        "env_p_error_start"             :env_p_error_interval_start,
        "env_p_error_final"             :env_p_error_interval_final,
        "env_p_error_delta"             :env_p_error_interval_increase,
        "env_p_error_strategy"          :env_p_error_strategy,
        "no_cuda_actors"                :no_cuda_actors,
        "no_cpu_actors"                 :no_cpu_actors,
        "log_actor"                     :log_priority_dist
    }

    # log header to tensorboard
    if could_import_tb:
        log("runs/{}/RunInfo/".format(learner_save_date), actor_args, learner_args, mem_args, state_dict_path)

    io_process = mp.Process(target=io, args=(mem_args,))
    actor_process = []    
    for i in range(actor_no_actors):
        if i < no_cuda_actors :
           actor_args["device"] = 'cuda'
        else: 
           actor_args["device"] = 'cpu'
        
        actor_args["id"] = i
        #actor_args["epsilon_final"] = epsilon[i * actor_no_envs : i * actor_no_envs + actor_no_envs]
        actor_args["epsilon_final"] = epsilon
        actor_process.append(mp.Process(target=actor, args=(actor_args,)))
        actor_process[i].start()
    
    io_process.start()
    try:
        learner(learner_args) 
    except:
        tb = SummaryWriter("runs/{}/RunInfo/".format(learner_save_date))
        tb.add_text("RunInfo/Error_Message", sys.exc_info()[0])
        tb.close()

    time.sleep(2)
    print("Training done.")
    for i in range(actor_no_actors):
        actor_process[i].terminate()
    io_process.terminate()
    print("Script complete.")
    
        
def log(path, actor, learner, memory, st_dict):
    tb_setup_string = (
                    "starting_params: {}  \n"
                    "env_size: {}  \n"
                    "learning_rate: {}  \n"
                    "learner_update_policy: {}  \n"
                    "learner_optimizer: {}  \n"
                    "learner_device: {}  \n"
                    "learner_job_max_time: {}  \n"
                    "learner_save_date: {}  \n"
                    "learner_eval_no_episodes: {}  \n"
                    "learner_eval_freq: {}  \n"
                    "actor_max_actions_per_episode: {}  \n"
                    "actor_size_local_memory_buffer: {}  \n"
                    "actor_no_envs: {}  \n"
                    "no_cuda_actors: {}  \n"
                    "no_cpu_actors: {}  \n"
                    "env_p_error_interval_start: {}  \n"   
                    "env_p_error_interval_final: {}  \n"   
                    "env_p_error_interval_increase: {}  \n"
                    "env_p_error_strategy: {}  \n"
                    "replay_memory_size: {}  \n"                            
                    "replay_memory_alpha: {}  \n"                 
                    "replay_memory_beta: {}  \n"                  
                    "replay_memory_size_before_sampeling: {}  \n" 
                    "replay_memory_batch_in_queue_limit: {}  \n"  
                    "log_priority_dist: {}  \n"                   
                    "log_write_frequency: {}  \n"                 
                    "log_priority_sample_max: {}  \n"             
                    "log_priority_sample_interval_size: {}  \n"
                    "batch_size: {}  \n"
                    "discount_factor: {}  \n").format(  st_dict,
                                                        learner["env_config"]["size"],
                                                        learner["learning_rate"],
                                                        learner["policy_update"],
                                                        learner["optimizer"],
                                                        learner["device"],
                                                        learner["job_max_time"],
                                                        learner["save_date"],
                                                        learner["learner_eval_no_episodes"],
                                                        learner["learner_eval_freq"],
                                                        actor["max_actions_per_episode"],
                                                        actor["size_local_memory_buffer"],
                                                        actor["no_envs"],
                                                        actor["no_cuda_actors"],
                                                        actor["no_cpu_actors"],
                                                        actor["env_p_error_start"],
                                                        actor["env_p_error_final"],
                                                        actor["env_p_error_delta"],
                                                        actor["env_p_error_strategy"],
                                                        memory["capacity"],
                                                        memory["alpha"],
                                                        memory["beta"],
                                                        memory["replay_size_before_sampling"],
                                                        memory["batch_in_queue_limit"],
                                                        memory["log_priority_dist"],
                                                        memory["log_write_frequency"],
                                                        memory["log_priority_sample_max"],
                                                        memory["log_priority_sample_interval_size"],
                                                        memory["batch_size"],
                                                        actor["discount_factor"])
    tb = SummaryWriter(path)
    tb.add_text("RunInfo/HyperParams", tb_setup_string)
    tb.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    start_distributed_mp()   
    
