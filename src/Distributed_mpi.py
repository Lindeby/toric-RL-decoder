from mpi4py import MPI
import torch
from src.nn.torch.ResNet import ResNet18
from src.Actor_mpi import Actor
from src.Learner_mpi import Learner
from src.ReplayMemory_mpi import ReplayMemory

# Setup
# Learner specific
learner_training_steps = 64
learner_learning_rate = 0.00025
learner_policy_update = 100
learner_optimizer = 'Adam'
learner_device = 'cuda'
learner_eval_freq = 100

# Actor specific
actor_max_actions_per_episode = 5 
actor_size_local_memory_buffer = 1000
actor_beta = 1 
actor_device = 'cpu'
actor_n_step = 1

# Replay Memory specific
replay_memory_size = 1000000 
replay_memory_alpha = 0.6
replay_memory_beta = 0.4
replay_memory_size_before_sampeling = 1000

# Shared
batch_size = 32
transition_priorities_discount_factor = 0.95
env = "toric-code-v0",
env_config = {  "size": 9,
                "min_qubit_errors": 0,
                "p_error": 0.1
        }
model = ResNet18
model_config = {"system_size": env_config["size"],
                "number_of_actions": 3
                }


base_comm = MPI.COMM_WORLD
base_rank = base_comm.Get_rank()
world_size = base_comm.Get_size()

# Every process reports to root if the have a gpu resorse or not. Root then
# decides which process is Learner, Memory Replay and Actors
base_comm_setup = None
cuda = torch.cuda.is_available()
cuda_available = base_comm.gather(cuda, root=0)

if base_rank == 0:

    learner = None
    memory = None
    for i in range(world_size):
        if cuda_available[i]:
            learner = i

    if learner < world_size-1:
        memory = learner +1
    else:
        memory = learner -1
    
    base_comm_setup = {
                    "learner":learner,
                    "memory":memory
                 }

base_comm_setup = base_comm.bcast(base_comm_setup, root=0)
     
# Create new comm group for actor/learner comunication 
exclude = []
exclude.append(base_comm_setup["memory"])
actor_learner_group = base_comm.group.Excl(exclude)
comm_actor_learner = base_comm.Create_group(actor_learner_group)

# Learner
if base_rank == base_comm_setup["learner"]:

    
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
        "mpi_base_comm"                 :base_comm,
        "mpi_comm_actor_learner"        :comm_actor_learner,
        "mpi_base_comm_setup"           :base_comm_setup
    }

    Learner(learner_args)
# Replay Memory
elif base_rank == base_comm_setup["memory"]:
    
    """
        Memory Process
    """
    mem_args = {
        "capacity"                          :replay_memory_size,
        "alpha"                             :replay_memory_alpha,
        "beta"                              :replay_memory_beta,
        "batch_size"                        :batch_size,
        "replay_size_before_sampling"       :replay_memory_size_before_sampeling if not (replay_memory_size_before_sampeling is None) else min(batch_size, int(replay_memory_size*0.1)),
        "mpi_base_comm"                     :base_comm,
        "mpi_base_comm_setup"               :base_comm_setup
    }

    ReplayMemory(mem_args)
          
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
        "mpi_base_comm"                 :base_comm,
        "mpi_comm_actor_learner"        :comm_actor_learner,
        "mpi_base_comm_setup"           :base_comm_setup
    }

    Actor(actor_args)
