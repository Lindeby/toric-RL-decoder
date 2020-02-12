# standard libraries
import os
from copy import deepcopy
from collections import namedtuple
# pytorch
from torch import from_numpy
import torch.distributed as dist
from torch.multiprocessing import Process, SimpleQueue, Pipe, Queue
# other files
from .learner import learner
from .actor import actor
from .DistributedReplayMemory import experienceReplayBuffer
import time
                
            
    

class Distributed():
        
    def __init__(self,  policy_net,
                        policy_config,
                        target_net,
                        target_config,
                        env,
                        env_config,
                        device, 
                        optimizer, 
                        replay_size, 
                        alpha, 
                        beta, 
                        memory_batch_size
                        ):

        self.policy_net = policy_net
        self.policy_config = policy_config
        self.target_net = target_net
        self.target_config = target_config
        
        self.env = env
        self.env_config = env_config

        self.optimizer = optimizer
        self.device = device
        self.replay_size = replay_size
        self.alpha = alpha
        self.beta = beta
        self.memory_batch_size = memory_batch_size
        
        if memory_batch_size > replay_size:
            raise ValueError("Please make sure replay memory size is larger than batch size.")
        
        # self.grid_shift = int(env.system_size/2)



    def train(self, training_steps,
                    no_actors, 
                    learning_rate, 
                    epsilons,
                    beta,
                    batch_size, 
                    policy_update, 
                    discount_factor,
                    max_actions_per_episode,
                    size_local_memory_buffer,
                    replay_size_before_sample = None
                    ):
        
        world_size = no_actors +2 #(+ Learner proces and Memmory process)
        actor_processes = []

        # Communication channels between processes
        transition_queue_to_memory = Queue()
        transition_queue_from_memory = Queue()
        update_priorities_queue_to_memory = Queue()

        # Communication pipes from learner to actors, one for each actor
        # For sending new network weights to the actors
        # The pipes are one way comunication (duplex = False)
        con_learner_actor = []
        con_actor_learner = []
        for a in range(no_actors):
            con_1, con_2 = Pipe(duplex=True)
            con_learner_actor.append(con_1)
            con_actor_learner.append(con_2)

        con_learner_memory, con_memory_learner = Pipe(duplex=True)



        """
            Learner Process
        """
        learner_args = {
            "no_actors"                            :no_actors,
            "train_steps"                          :training_steps,
            "batch_size"                           :batch_size,
            "learning_rate"                        :learning_rate,
            "policy_update"                        :policy_update,
            "discount_factor"                      :discount_factor,
            "optimizer"                            :self.optimizer,
            "policy_net"                           :self.policy_net,
            "policy_config"                        :self.policy_config,
            "target_net"                           :self.target_net,
            "target_config"                        :self.target_config,
            "device"                               :self.device,
            "transition_queue_from_memory"         :transition_queue_from_memory,
            "update_priorities_queue_to_memory"    :update_priorities_queue_to_memory,
            "con_actors"                           :con_learner_actor,
            "con_replay_memory"                    :con_learner_memory,
            "env"                                  :self.env,
            "env_config"                           :self.env_config
        }

         
        learner_process = Process(target=self._init_process, 
                                  args=(0, 
                                        world_size, 
                                        learner, 
                                        learner_args))
        learner_process.start()
        #processes.append(learner_process)
        
        """
            Memory Process
        """
        mem_args = {
            "capacity"                          :self.replay_size,
            "alpha"                             :self.alpha,
            "beta"                              :self.beta,
            "batch_size"                        :self.memory_batch_size,
            "transition_queue_to_memory"        :transition_queue_to_memory,
            "transition_queue_from_memory"      :transition_queue_from_memory,
            "update_priorities_queue_to_memory" :update_priorities_queue_to_memory,
            "con_learner"                       :con_memory_learner,
            "replay_size_before_sampling"       :batch_size if not None else min(batch_size, self.replay_size*0.25)
            }
        
        print("Memory Process")
        memmory_process = Process(target = self._init_process,
                                  args=(1, 
                                        world_size,
                                        experienceReplayBuffer,
                                        mem_args))

        memmory_process.start()
        #processes.append(memmory_process)

        """
            Actor Processes
        """
        actor_args = { 
            "train_steps"                   :training_steps, 
            "max_actions_per_episode"       :max_actions_per_episode, 
            "update_policy"                 :policy_update, 
            "size_local_memory_buffer"      :size_local_memory_buffer, 
            "env_config"                    :self.env_config,
            "model"                         :self.policy_net,
            "model_config"                  :self.policy_config,
            "env"                           :self.env,
            "device"                        :self.device,
            "beta"                          :beta,
            "discount_factor"               :discount_factor,
            "transition_queue_to_memory"    :transition_queue_to_memory
            }
    
        for rank in range(no_actors):
            actor_args["epsilon"] = epsilons[rank]
            actor_args["con_learner"] = con_actor_learner[rank] 
            
            actor_process = Process(target=self._init_process, 
                                    args=(rank+2, 
                                          world_size, 
                                          actor, 
                                          actor_args))

            actor_process.start()
            print("starting actor ",(rank + 2))
            actor_processes.append(actor_process)

        for a in actor_processes:
            a.join()
            print(a, "joined")

        memmory_process.join()
        learner_process.join()

    def _init_process(self, rank, size, fn, args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.2'
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, args)
        
    

        

