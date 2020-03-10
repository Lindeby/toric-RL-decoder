import torch
from src.nn.torch.ResNet import ResNet18
from src.nn.torch.NN import NN_11, NN_17
from src.Actor_mp import actor
from src.IO_mp import io
from src.Learner_mp import learner
from src.util_actor import calculateEpsilon
import numpy as np
import multiprocessing as mp

from datetime import datetime


def start_distributed_mp():

    # Setup
    
    
    
    # Learner specific
    learner_training_steps = 10000000
    learner_learning_rate = 0.00025
    learner_policy_update = 100
    learner_optimizer = 'Adam'
    learner_device = 'cuda'
    learner_eval_freq = 100
    learner_synchronize = 50
    learner_job_max_time =  200#60*3 -2 #2 hours 58min
    learner_save_date = datetime.now().strftime("%d_%b_%Y_%H_%M_%S") 
   
    # Actor specific
    actor_max_actions_per_episode = 25 
    actor_size_local_memory_buffer = 10
    actor_beta = 1 
    actor_device = 'cpu'
    actor_n_step = 1
    actor_no_envs = 2           #number of envs/actor
    actor_no_actors = 1
    epsilon = calculateEpsilon(0.8, 7, actor_no_actors * actor_no_envs)
    
    # Replay Memory specific
    replay_memory_size = 1000000 
    replay_memory_alpha = 0.6
    replay_memory_beta = 0.4
    replay_memory_size_before_sampeling = 100#replay_memory_size * 0.05
    replay_memory_batch_in_queue_limit = 10 #number of batches in queue to learner
    
    # Shared
    batch_size = 32
    transition_priorities_discount_factor = 0.95
    env = "toric-code-v0"
    env_config = {  "size": 9,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
            }
    #model = ResNet18
    model = NN_11
    model_config = {"system_size": env_config["size"],
                    "number_of_actions": 3
                    }
    
    #Comm setup 
    actor_io_queue = mp.Queue()
    learner_io_queue = mp.Queue()
    io_learner_queue = mp.Queue()
    
    pipe_io_actor = []
    pipe_actor_io = []
    for i in range(actor_no_actors):
        con1, con2 = mp.Pipe(duplex=True)
        pipe_io_actor.append(con1)
        pipe_actor_io.append(con2)

    pipe_learner_io, pipe_io_learner = mp.Pipe(duplex=True)

    
    """
        Learner Process
    """
    learner_args = {
        "train_steps"                   :learner_training_steps,
        "batch_size"                    :batch_size,
        "learning_rate"                 :learner_learning_rate,
        "policy_update"                 :learner_policy_update,
        "discount_factor"               :transition_priorities_discount_factor,
        "optimizer"                     :learner_optimizer,
        "model"                         :model,
        "model_config"                  :model_config,
        "device"                        :learner_device,
        "eval_freq"                     :learner_eval_freq,
        "env"                           :env,
        "env_config"                    :env_config,
        "synchronize"                   :learner_synchronize,
        "job_max_time"                  :learner_job_max_time,
        "save_date"                     :learner_save_date,
        "learner_io_queue"              :learner_io_queue,
        "io_learner_queue"              :io_learner_queue,
        "pipe_learner_io"               :pipe_learner_io
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
        "pipe_io_actor"                     :pipe_io_actor,
        "pipe_io_learner"                   :pipe_io_learner,
        "batch_in_queue_limit"              :replay_memory_batch_in_queue_limit,
        "no_actors"                         :actor_no_actors,
        "replay_size_before_sampling"       :replay_memory_size_before_sampeling if not (replay_memory_size_before_sampeling is None) else min(batch_size, int(replay_memory_size*0.1)),
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
        "env"                           :env,
        "device"                        :actor_device,
        "beta"                          :actor_beta,
        "discount_factor"               :transition_priorities_discount_factor,
        "n_step"                        :actor_n_step,
        "no_envs"                       :actor_no_envs,
        "actor_io_queue"                :actor_io_queue
    }

    io_process = mp.Process(target=io, args=(mem_args,))
    #learner_process = mp.Process(target=learner, args=(learner_args,))
    actor_process = []
    
    for i in range(actor_no_actors):
        actor_args["epsilon"] = epsilon[i * actor_no_envs : i * actor_no_envs + actor_no_envs]
        actor_args["pipe_actor_io"] = pipe_actor_io[i] 
        actor_process.append(mp.Process(target=actor, args=(actor_args,)))
        actor_process[i].start()
    
    io_process.start()
    #learner_process.start()
    learner(learner_args) 

    for i in range(actor_no_actors):
        actor_process[i].join()

    #learner_process.join()
    io_process.join()
    
        

if __name__ == '__main__':
    mp.set_start_method('spawn')
    start_distributed_mp()   
    
