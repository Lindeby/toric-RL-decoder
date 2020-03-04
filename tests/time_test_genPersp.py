import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.numba.util import generatePerspectiveOptimized as numba_gpo
from src.util import generatePerspectiveOptimized as gpo
import gym, gym_ToricCode
import numpy as np
import time

if __name__ == "__main__":
    SIZE = 3
    env_config = {  
        "size": SIZE,
        "min_qubit_errors": 0,
        "p_error": 0.1
    }

    env = gym.make('toric-code-v0', config=env_config)


    state  = env.reset()
    # compile call
    numba_gpo(int(SIZE/2), SIZE, state)

    for i in range(1000):
        state  = env.reset()

        start = time.time()
        res0 = gpo(int(SIZE/2), SIZE, state)
        mid = time.time()
        res1 = numba_gpo(int(SIZE/2), SIZE, state)
        end = time.time()

        for p in range(len(res0)):
            if not np.all(np.equal(res0[p], res1[p])):
                print(res0[p])
                print("")
                print(res1[p])
                exit()

        native  = mid-start
        numb    = end-mid
        print("Native: {}ms   Numba: {}, (Native - Numba): {}ms".format(native*1000, numb*1000, (native-numb)*1000))

    print("All tests ran successfully.")
