import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.numba.util import generatePerspectiveOptimized
from src.numba.util_actor import selectActionParallel as numba_sap
from src.util_actor import selectActionParallel as sap
from src.nn.torch.ResNet import ResNet18
from src.EnvSet import EnvSet

import gym, gym_ToricCode, time
import numpy as np

if __name__ == "__main__":
    SIZE = 3
    NO_ENVS = 50

    env_config = {  
        "size": SIZE,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)
    envs = EnvSet(env, NO_ENVS)

    model = ResNet18()

    states  = envs.resetAll()
    # compile call
    numba_sap(3, 0, int(SIZE/2), SIZE, states, model, 'cpu')


    for i in range(10000):

        states = envs.resetAll()

        start = time.time()
        a0, q0  =  sap(3, 0, int(SIZE/2), SIZE, states, model, 'cpu')
        mid   = time.time()
        a1, q1  = numba_sap(3, 0, int(SIZE/2), SIZE, states, model, 'cpu')
        end   = time.time()

        q_diff = np.absolute(q0-q1)
        if not np.all(np.equal(a0, a1)) or np.all(q_diff > 1e-8):
            print(a0[i], a1[i])
            print(q0[i], q1[i])
            exit()

        native  = mid-start
        numb    = end-mid
        print("Native: {}ms   Numba: {}, (Native - Numba): {}ms".format(native*1000, numb*1000, (native-numb)*1000))

    print("All tests ran successfully.")
