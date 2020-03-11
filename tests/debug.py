import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from multiprocessing import Process, set_start_method
from multiprocessing.sharedctypes import Array, copy, Value
from src.nn.torch.ResNet import ResNet18
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from torch import from_numpy
import time

def worker(rank, args):
    X = args["X"]
    Y = args["Y"]

    if rank == 0:
        for i in range(20):
            time.sleep(.8)
            
            with X.get_lock():
                print("Pusher got lock on X.")
                X[0] = i
                Y.value = i
    else:
        idx = 0
        old_idx = 0
        model = ResNet18()
        X_new = np.empty(args["X_len"])
        print("Worker {} starting loop.".format(rank))
        while True:
            with X.get_lock():
                # print("Worker {} locked X, Y is {}.".format(rank, Y.value))
                if idx < Y.value:
                    idx = Y.value
                    X_np = np.frombuffer(X.get_obj())
                    np.copyto(X_new, X_np)
                
                if idx != old_idx:
                    # print("Worker {} updated weights.".format(rank))
                    #print("Worker {} first params: {}".format(rank, X_new[:7]))
                    vector_to_parameters(from_numpy(X_new), model.parameters())
                    old_idx = idx
            print("Worker {} first params: {}".format(rank, X_new[:7]))
            if idx >= 7:
                break

            time.sleep(.5*rank)






if __name__ == "__main__":
    set_start_method('spawn')

    model = ResNet18()

    params = parameters_to_vector(model.parameters()).detach().numpy()
    X = Array('d', len(params))
    Y = Value('i')
    Y.value = 0

    X_np = np.frombuffer(X.get_obj())

    np.copyto(X_np, params)

    args = {    "X":X,
                "X_len": len(params),
                "Y": Y
            }

    size = 3
    processes = []
    for i in range(size):
        p = Process(target=worker, args=(i, args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()