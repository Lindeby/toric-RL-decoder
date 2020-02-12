from copy import deepcopy
from src.Distributed import Distributed
#from gym_ToricCode import gym_ToricCode 
import gym, torch
import gym_ToricCode

from src.nn.torch.NN import NN_11, NN_17
from src.nn.torch.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# valid network names: 
#   NN_11
#   NN_17
#   ResNet18
#   ResNet34
#   ResNet50
#   ResNet101
#   ResNet152

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# common system sizes are 3,5,7 and 9 
# grid size must be odd! 
SYSTEM_SIZE = 3 
MIN_QBIT_ERRORS = 0
P_ERROR = 0.1

NETWORK = NN_17

env_config = {  "size": SYSTEM_SIZE,
                "min_qubit_errors": MIN_QBIT_ERRORS,
                "p_error": P_ERROR
        }

model_config = {"system_size": env_config["size"],
                "number_of_actions": 3
                }


dl = Distributed(policy_net = NETWORK,
                 policy_config = model_config,
                 target_net = NETWORK,
                 target_config = model_config,
                 env = "toric-code-v0",
                 env_config = env_config,
                 device = device,
                 optimizer  = 'Adam',
                 replay_size= 100,
                 alpha = 0.6,
                 beta = 0.4
                )

epsilons = [0.9, 0.5]

dl.train(training_steps = 100,
        no_actors = 1,
        learning_rate = 0.00025,
        epsilons = epsilons,
        beta = 1,
        batch_size = 16,
        policy_update = 10,
        discount_factor = 0.9,
        max_actions_per_episode = 10,
        size_local_memory_buffer = 10,
        eval_freq=10
        )





                  
                        

