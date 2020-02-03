from torch.multiprocessing import Process
import torch.distributed as dist
from src.nn.torch.NN import NN_11
import gym, gym_ToricCode
from collections import namedtuple
import os
import numpy as np
from torch import from_numpy
import torch
from src.actor import generatePerspective 
import time


Perspective = namedtuple('Perspective', ['perspective', 'position'])
# https://pytorch.org/docs/stable/notes/multiprocessing.html

def worker(rank, size, model):

    if rank == 0:
        while True: continue
        # Send weights
    else:
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



def init_process(rank, size, fn, model, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, model)


model = NN_11(3, 3, 'cpu')

size = 2
processes = []
for i in range(size):
    p = Process(target=init_process, args=(i, size, worker, model))
    processes.append(p)
    p.start()

for p in processes:
    p.join()