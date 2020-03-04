import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.numba.util import generatePerspectiveOptimized as numba_gpo
from src.util import generatePerspectiveOptimized as gpo
import gym, gym_ToricCode
import numpy as np

if __name__ == "__main__":
    env_config = {  
        "size": 3,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)

    state  = env.reset()
    # TODO:CONTINUE