from copy import deepcopy
from src.Distributed import Distributed
#from gym_ToricCode import gym_ToricCode 
import gym, torch
import gym_ToricCode
from torch.multiprocessing import set_start_method
import torchvision.models as models

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

if __name__ == "__main__":
        
        set_start_method('spawn')


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # common system sizes are 3,5,7 and 9 
        # grid size must be odd! 
        SYSTEM_SIZE = 3
        MIN_QBIT_ERRORS = 0
        P_ERROR = 0.1

        NETWORK = ResNet18

        env_config = {  "size": SYSTEM_SIZE,
                        "min_qubit_errors": MIN_QBIT_ERRORS,
                        "p_error": P_ERROR
                }

        model_config = {"system_size": env_config["size"],
                        "number_of_actions": 3
                        }


        dl = Distributed(policy_net = NETWORK,
                         policy_config = model_config,
                         env = "toric-code-v0",
                         env_config = env_config,
                         device = device,
                         optimizer  = 'Adam',
                         replay_size= 10000,
                         alpha = 0.6,
                         beta = 0.4
                        )

        epsilons = [0.3]

        dl.train(training_steps = 1001,
                no_actors = 1,
                learning_rate = 0.00025,
                epsilons = epsilons,
                beta = 1,
                n_step = 1,
                batch_size = 32,
                policy_update = 100,
                discount_factor = 0.95,
                max_actions_per_episode = 75,
                size_local_memory_buffer = 1000,
                eval_freq=100,
                replay_size_before_sample = 1000
                )






                        

