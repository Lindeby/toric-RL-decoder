from copy import deepcopy
from src.Distributed import Distributed
from gym_ToricCode import gym_ToricCode 
import gym, torch

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

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

NETWORK = NN_17(SYSTEM_SIZE, 3, 'cpu')

#gym.make('toric-code-v0', size = SYSTEM_SIZE, min_qbit_errors = MIN_QBIT_ERRORS, p_error = P_ERROR)
toric_enviroment = ToricCodeEnv(SYSTEM_SIZE, MIN_QBIT_ERRORS, P_ERROR)

dl = Distributed(policy_net = NETWORK,
                 target_net = deepcopy(NETWORK),
                 device = device,
                 optimizer  = 'Adam',
                 replay_size= 100,
                 alpha = 0.6,
                 env = toric_enviroment
                 )

epsilons = [0.1]

dl.train(training_steps = 50,
        no_actors = 1,
        learning_rate = 0.00025,
        epsilons = epsilons,
        batch_size = 16,
        policy_update = 100,
        discount_factor = 0.9)





                  
                        

