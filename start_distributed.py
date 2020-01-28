from copy import deepcopy
from src.Distributed import Distributed
import gym, gym_ToricCode


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
NETWORK = NN_17

# common system sizes are 3,5,7 and 9 
# grid size must be odd! 
SYSTEM_SIZE = 3 
MIN_QBIT_ERRORS = 0
P_ERROR = 0.1

toric_enviroment = gym.make('toric-code-v0', size = SYSTEM_SIZE, min_qbit_errors = MIN_QBIT_ERRORS, p_error = P_ERROR)
# toric_enviroment.__init__(size = SYSTEM_SIZE, min_qbit_errors = MIN_QBIT_ERRORS, p_error = P_ERROR)
dl = Distributed(policy_net = NETWORK,
                 target_net = deepcopy(NETWORK),
                 optimizer  = 'Adam',
                 env = toric_enviroment,
                 replay_memory = 'proportional'
                 )

#epsilons = [0.1, 0.5]
#
#dl.train(training_steps = 50,
#         no_actors = 2,
#         learning_rate = 0.00025,
#         batch_size = 64,
#         policy_update = 100)




                  
                        

