from src.nn.torch.NN import NN_11, NN_17
from collections import namedtuple
import os
import time
import numpy as np
import torch
from torch import from_numpy
import torch.distributed as dist
from torch.multiprocessing import Process, SimpleQueue
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.actor import generatePerspective 
import gym, gym_ToricCode


Perspective = namedtuple('Perspective', ['perspective', 'position'])
# https://pytorch.org/docs/stable/notes/multiprocessing.html

def worker(rank, size, args):
    model = args["model"]
    q = args["q"]

    if rank == 0:
        weights = model.state_dict()
        for a in range(size-1):
            q.put(weights)
            print("Learner put weight in queue {} times.".format(a))
        
        time.sleep(2)

    else:
        weights = None
        while True:
            if not q.empty():
                weights = q.get()
                break

        print("Rank {} loading weights.".format(rank))
        model.load_state_dict(weights)
        print("Rank {} loaded weights successfully.".format(rank))
        
        env = gym.make('toric-code-v0', config={})
        state = env.reset()

        perspectives = generatePerspective(1,3, state)

        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = from_numpy(batch_perspectives).type('torch.Tensor')    
        batch_perspectives = batch_perspectives.to('cpu')

        print("Starting prediction...")
        with torch.no_grad():
            policy_net_output = model(batch_perspectives)
            print(policy_net_output)



def init_process(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


q = SimpleQueue()
model = NN_17(3, 3, 'cpu')

args = {    "q":q,
            "model":model
        }

size = 3
processes = []
for i in range(size):
    p = Process(target=init_process, args=(i, size, worker, args))
    processes.append(p)
    p.start()

for p in processes:
    p.join()