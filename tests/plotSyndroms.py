import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

import gym, gym_ToricCode
import numpy as np
from src.util_actor import generatePerspective

if __name__ == "__main__":

    s = [[[0,0,0,0,0]
        , [0,0,0,0,0]
        , [0,0,3,0,0]
        , [0,0,3,0,0]
        , [0,0,0,0,0]],

        [ [0,0,0,0,0]
        , [3,3,0,0,0]
        , [0,0,0,0,0]
        , [0,0,0,0,0]
        , [0,0,0,0,0]]]

    start_error = np.array(s) 

    env_config = {  
        "size": start_error.shape[-1],
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)

    env.reset()


    state = env.createSyndromOpt(start_error)
    env.qubit_matrix = start_error
    env.state = state

    env.plotToricCode(state, 'perspective_'+'2')

