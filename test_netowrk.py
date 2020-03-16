# torch
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy

import gym, gym_ToricCode
# python lib
import numpy as np 
import random
from copy import deepcopy
# from file 
from src.util import Action, Perspective, Transition, generatePerspective, rotate_state, shift_state
from src.util_actor import updateRewards, selectAction, computePriorities, generateTransition
# Quality of life
from src.nn.torch.NN import NN_11, NN_17
from src.nn.torch.ResNet import ResNet18

     
device = 'cpu'

env = "toric-code-v0"
env_config = {  "size": 3,
                "min_qubit_errors": 0,
                "p_error": 0.1
        }
model = NN_11
model_config = {"system_size": env_config["size"],
                "number_of_actions": 3
                }
max_actions = 25
NN = model
if NN == NN_11 or NN == NN_17:
    NN_config = model_config
    model = NN(NN_config["system_size"], NN_config["number_of_actions"], device)
else:
    model = NN()

model.load_state_dict(torch.load('Size_3_NN_11_14_Mar_2020_04_39_05.pt', map_location=torch.device('cpu')))    
model.to(device)
model.eval()

# env and env params
env = gym.make(env, config=env_config)

no_actions = int(env.action_space.high[-1])
grid_shift = int(env.system_size/2)

# startup
state = env.reset()
steps_per_episode = 0

# main loop over training steps
while True:

    env.plotToricCode(state, "eval") 
    input("Press key to step") 
    steps_per_episode += 1    
    # select action using epsilon greedy policy
    action, q_values = selectAction(number_of_actions=no_actions,
                                    epsilon=0, 
                                    grid_shift=grid_shift,
                                    toric_size = env.system_size,
                                    state = state,
                                    model = model,
                                    device = device)

    next_state, reward, terminal_state, _ = env.step(action)

    
    if terminal_state or steps_per_episode > max_actions:
        if terminal_state:
            print("Solved")
        else:
            print("to many actions")
        # reset env
         
        env.plotToricCode(state, "eval") 
        input("Press key to reset") 
        state = env.reset()
        steps_per_episode = 0
    else:
        state = next_state

