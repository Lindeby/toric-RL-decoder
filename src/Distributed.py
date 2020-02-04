# standard libraries
import os
from copy import deepcopy
from collections import namedtuple
from .ReplayMemory import PrioritizedReplayMemory
# pytorch
from torch import from_numpy
import torch.distributed as dist
from torch.multiprocessing import Process, SimpleQueue
# other files
from .learner import learner
from .actor import actor
from .ReplayMemory import PrioritizedReplayMemory


class Distributed():
    
    Transition = namedtuple('Transition',['previous_state', 
                                          'action', 
                                          'reward', 
                                          'state', 
                                          'terminal']) 
    
    
    def __init__(self, policy_net, target_net, device, optimizer, replay_size, alpha, env):

        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.replay_size = replay_size
        self.alpha = alpha
        self.policy_net = policy_net
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = target_net
        self.target_net = self.target_net.to(self.device)

        self.replay_memory = PrioritizedReplayMemory(replay_size, alpha) # TODO: temp size, alpha
        

    def train(self, training_steps, no_actors, learning_rate, epsilons, batch_size, policy_update, discount_factor):
        print("start training")
        world_size = no_actors +1
        processes = []

        # Communication channels between processes
        weight_queue = SimpleQueue()
        transition_queue = SimpleQueue()
        transition_queue_to_memory = SimpleQueue()
        transition_queue_from_memory = SimpleQueue()
        update_priorities_queue_to_memory = SimpleQueue()


        #args = {"capacity":self.replay_size,
        #        "alpha": self.alpha,
        #        }
        #
        #memmory_process = Process(target = self._init_process,
        #                          args=(1, 
        #                                world_size,
        #                                eperienceReplayBuffer,
                                        
                                        
          
        
        args = {"no_actors": no_actors,
                "train_steps":training_steps,
                "batch_size":batch_size,
                "optimizer":self.optimizer,
                "policy_net":self.policy_net,
                "target_net": self.target_net,
                "learning_rate":learning_rate,
                "device":self.device,
                "policy_update":policy_update,
                "replay_memory":self.replay_memory,
                "discount_factor":discount_factor,
                "transition_queue":transition_queue,
                "weight_queue":weight_queue
                }
         
        learner_process = Process(target=self._init_process, 
                                  args=(0, 
                                        world_size, 
                                        learner, 
                                        args))
        learner_process.start()
        processes.append(learner_process)


        args = {"train_steps": training_steps, 
                "max_actions_per_episode":5, 
                "update_policy":policy_update,
                "size_local_memory_buffer":50, 
                "min_qubit_errors":0, 
                "model":self.policy_net,
                "env":self.env,
                "device":self.device,
                "beta": 1,
                "discount_factor":discount_factor,
                "transition_queue":transition_queue,
                "weight_queue":weight_queue
                }
    
        for rank in range(no_actors):
            args["epsilon"] = epsilons[rank]
            
            actor_process = Process(target=self._init_process, 
                                    args=(rank+1, 
                                          world_size, 
                                          actor, 
                                          args))

            actor_process.start()
            print("starting actor ",(rank + 1))
            processes.append(actor_process)

        for p in processes:
            p.join()
            print(p, "joined")

    def _init_process(self, rank, size, fn, args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.2'
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, args)
        
    
    def eperienceReplayBuffer(rank, 
                              worl_size, 
                              args):
        
        
        args = {"capacity",
                "alpha"}
       
        memory = PrioritizedReplayMemory(capacity, alpha)

