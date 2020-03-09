import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.numba.util import generatePerspectiveOptimized
from src.numba.util_actor import selectActionParallel as numba_sap
from src.numba.util_actor import generateTransitionParallel as numba_gtp
from src.util import action_type
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

    transition_type = np.dtype([('perspective', (np.int, (2,SIZE,SIZE))),
                                ('action', action_type),
                                ('reward', np.float),
                                ('next_perspective', (np.int, (2,SIZE,SIZE))),
                                ('terminal',np.bool)])


    model = ResNet18()

    states  = envs.resetAll()
    # compile call
    a, q = numba_sap(3, 0, int(SIZE/2), SIZE, states, model, 'cpu')
    next_state, reward, terminal, _ = envs.step(a)
    transitions = numba_gtp(a, reward, states, next_state, terminal, int(SIZE/2), transition_type)

    for i in range(100):

        states = envs.resetAll()
        a, q  = numba_sap(3, 0, int(SIZE/2), SIZE, states, model, 'cpu')

        next_state, reward, terminal, _ = envs.step(a)

        start = time.time()
        mid   = time.time()
        transitions = numba_gtp(a, reward, states, next_state, terminal, int(SIZE/2), transition_type)
        end   = time.time()

        # q_diff = np.absolute(q0-q1)
        # if not np.all(np.equal(a0, a1)) or np.all(q_diff > 1e-8):
        #     print(a0[i], a1[i])
        #     print(q0[i], q1[i])
        #     exit()

        native  = mid-start
        numb    = end-mid
        print("Native: {}ms   Numba: {}, (Native - Numba): {}ms".format(native*1000, numb*1000, (native-numb)*1000))

    print("All tests ran successfully.")
