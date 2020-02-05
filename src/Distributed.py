# standard libraries
import os
from copy import deepcopy
from collections import namedtuple
from .ReplayMemory import PrioritizedReplayMemory
# pytorch
from torch import from_numpy
import torch.distributed as dist
from torch.multiprocessing import Process, SimpleQueue, Pipe
# other files
from .learner import learner
from .actor import actor
from .ReplayMemory import PrioritizedReplayMemory


def experienceReplayBuffer(rank, 
                          worl_size, 
                          args):
    """ 
        args = {"capacity",
                "alpha",
                "beta",
                "batch_size",
                "transition_queue_to_memory",
                "transition_queue_form_memory",
                "update_priorities_queue_to_memory",
                }
    """
   
    transition_queue_to_memory = args["transition_queue_to_memory"]
    transition_queue_from_memory = args["transition_queue_from_memory"]
    update_priorities_queue_to_memory = args["update_priorities_queue_to_memory"]
    capacity = args["capacity"]
    alpha = args["alpha"]
    beta = args["beta"]
    batch_size = args["batch_size"]
    memory = PrioritizedReplayMemory(capacity, alpha)

    while(True):

        #Receive transitions from actors
        for _ in range(100):
            if transition_queue_to_memory.empty():
                break
            
            transition, priority = transition_queue_to_memory.get()
            memory.save(transition,priority)

        #Sample batch of transitions to learner
        for _ in range(10):
            transition, _, indices = memory.sample(batch_size, beta)
            transition_queue_from_memory.put([transition, indices])

        for _ in range(10):
            if update_priorities_queue_to_memory.empty():
                break
            
            indices, priorities = update_priorities_queue_to_memory.get()
            memory.priority_update(indices, priorities)
            


class Distributed():
    
    Transition = namedtuple('Transition',['previous_state', 
                                          'action', 
                                          'reward', 
                                          'state', 
                                          'terminal']) 
    
    
    def __init__(self, policy_net, target_net, device, optimizer, replay_size, alpha, beta, memory_batch_size, env):

        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.replay_size = replay_size
        self.alpha = alpha
        self.beta = beta
        self.memory_batch_size = memory_batch_size
        self.policy_net = policy_net
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = target_net
        self.target_net = self.target_net.to(self.device)

        self.replay_memory = PrioritizedReplayMemory(replay_size, alpha) # TODO: temp size, alpha
        

    def train(self, training_steps, no_actors, learning_rate, epsilons, batch_size, policy_update, discount_factor):
        world_size = no_actors +2 #(+ Learner proces and Memmory process)
        processes = []

        # Communication channels between processes
        transition_queue = SimpleQueue()
        transition_queue_to_memory = SimpleQueue()
        transition_queue_from_memory = SimpleQueue()
        update_priorities_queue_to_memory = SimpleQueue()

        # Communication pipes from learner to actors, one for each actor
        # For sending new network weights to the actors
        # The pipes are one way comunication (duplex = False)
        con_recive_weights = []
        con_send_weights = []
        for a in range(no_actors):
            con_receive, con_send = Pipe(duplex=False)
            con_send_weights.append(con_send)
            con_recive_weights.append(con_receive)



        
        """
            Learner Process
        """

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
                "transition_queue_from_memory":transition_queue_from_memory,
                "update_priorities_queue_to_memory":update_priorities_queue_to_memory,
                "con_send_weights":con_send_weights
                }
         
        learner_process = Process(target=self._init_process, 
                                  args=(0, 
                                        world_size, 
                                        learner, 
                                        args))
        learner_process.start()
        processes.append(learner_process)
        
        """
            Memory Process
        """
        args = {"capacity":self.replay_size,
                "alpha": self.alpha,
                "beta":self.beta,
                "batch_size":self.memory_batch_size,
                "transition_queue_to_memory":transition_queue_to_memory,
                "transition_queue_form_memory":transition_queue_from_memory,
                "update_priorities_queue_to_memory":update_priorities_queue_to_memory
                }
        
        print("Memory Proces")
        memmory_process = Process(target = self._init_process,
                                  args=(1, 
                                        world_size,
                                        experienceReplayBuffer,
                                        args))

        memmory_process.start()
        processes.append(memmory_process)

        """
            Actor Processes
        """

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
                "transition_queue":transition_queue
                }
    
        for rank in range(no_actors):
            args["epsilon"] = epsilons[rank]
            args["con_receive_weights"] = con_recive_weights[rank] 
            
            actor_process = Process(target=self._init_process, 
                                    args=(rank+2, 
                                          world_size, 
                                          actor, 
                                          args))

            actor_process.start()
            print("starting actor ",(rank + 2))
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
        
    

        

