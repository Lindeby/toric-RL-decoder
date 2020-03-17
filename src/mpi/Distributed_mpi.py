from mpi4py import MPI
import torch
from src.nn.torch.ResNet import ResNet18
from src.nn.torch.NN import NN_11, NN_17
from src.Actor_mpi import actor
from src.IO_mpi import io
from src.util_actor import calculateEpsilon
import numpy as np

from datetime import datetime


def start_distributed_mpi():

    # Setup

    # Mpi communication   
    base_comm = MPI.COMM_WORLD
    base_rank = base_comm.Get_rank()
    world_size = base_comm.Get_size()

    # Learner specific
    learner_training_steps = 10000000
    learner_learning_rate = 0.00025
    learner_policy_update = 100
    learner_optimizer = 'Adam'
    learner_device = 'cuda'
    learner_eval_freq = 100
    learner_synchronize = 50
    learner_job_max_time =  40#60*3 -2 #2 hours 58min
    learner_save_date = datetime.now().strftime("%d_%b_%Y_%H_%M_%S") 
   
    # Actor specific
    actor_max_actions_per_episode = 25 
    actor_size_local_memory_buffer = 15
    actor_beta = 1 
    actor_device = 'cpu'
    actor_n_step = 1
    actor_no_envs = 10           #number of envs/actor
    epsilon = calculateEpsilon(0.8, 7, (world_size -1)*actor_no_envs)
    
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
    env_config = {  "size": 3,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
            }
    #model = ResNet18
    model = NN_11
    model_config = {"system_size": env_config["size"],
                    "number_of_actions": 3
                    }
    
    
    
    # Every process reports to root if the have a gpu resorse or not. Root then
    # decides which process is Learner, Memory Replay and Actors
    learner_rank = None
    cuda = torch.cuda.is_available()
    cuda_available = base_comm.gather(cuda, root=0)
    if base_rank == 0:
        for i in range(world_size):
            if cuda_available[i]:
                learner_rank = i
                break 
    
    learner_rank = base_comm.bcast(learner_rank, root=0)

    # Learner
    if base_rank == learner_rank:
    
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
            "save_date"                     :learner_save_date
        }
        """
            Memory Process
        """
        mem_args = {
            "capacity"                          :replay_memory_size,
            "alpha"                             :replay_memory_alpha,
            "beta"                              :replay_memory_beta,
            "batch_size"                        :batch_size,
            "mpi_base_comm"                 :base_comm,
            "mpi_learner_rank"              :learner_rank,
            "batch_in_queue_limit"              :replay_memory_batch_in_queue_limit,
            "replay_size_before_sampling"       :replay_memory_size_before_sampeling if not (replay_memory_size_before_sampeling is None) else min(batch_size, int(replay_memory_size*0.1)),
        }
    
        io(mem_args, learner_args)
              
    # Actor
    else:
    
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
            "mpi_base_comm"                 :base_comm,
            "mpi_learner_rank"              :learner_rank
        }

        # give actor array of epsilons form epsion
        if learner_rank < base_rank:
            actor_args["epsilon"] = epsilon[base_rank-1]
            actor_args["epsilon"] = epsilon[(base_rank-1) * actor_no_envs : (base_rank-1) * actor_no_envs + actor_no_envs]
        else:
            actor_args["epsilon"] = epsilon[base_rank * actor_no_envs : base_rank * actor_no_envs + actor_no_envs]   
    
        actor(actor_args)
