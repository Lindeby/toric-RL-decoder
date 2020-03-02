# standard libraries
import os
from collections import namedtuple
from datetime import datetime
# pytorch
import torch.distributed as dist
from torch.multiprocessing import Process, Pipe, Queue, set_start_method
# other files
from .learner import learner
from .actor import actor
from .buffer import experienceReplayBuffer



class Distributed():
        
    def __init__(self,  policy_net,
                        policy_config,
                        env,
                        env_config,
                        device, 
                        optimizer, 
                        replay_size, 
                        alpha, 
                        beta, 
                        update_tb = 10
                        ):

        self.policy_net = policy_net
        self.policy_config = policy_config
        
        self.env = env
        self.env_config = env_config

        self.optimizer = optimizer
        self.device = device
        self.replay_mem_size = replay_size
        self.alpha = alpha
        self.beta = beta        
        
        self.update_tb = update_tb
        self.tb_log_dir = "runs/{}".format(datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))



    def train(self, training_steps,
                    no_actors, 
                    learning_rate, 
                    epsilons,
                    n_step,
                    beta,
                    batch_size, 
                    policy_update, 
                    discount_factor,
                    max_actions_per_episode,
                    size_local_memory_buffer,
                    eval_freq,
                    replay_size_before_sample = None,
                    no_envs = 1
                    ):
        

        if batch_size > self.replay_mem_size:
            raise ValueError("Please make sure replay memory size is larger than batch size.")
        if 1 > n_step:
            raise ValueError("Please have n_step >= 1.")
        if 1 >= size_local_memory_buffer:
            raise ValueError("Please let size_local_memory_buffer > 1.")
        if not isinstance(epsilons, list):
            raise ValueError("Please provide epsilons as a list.")
        if len(epsilons) != no_envs*no_actors:
            raise ValueError("Mismatch in epsilons and no_envs*no_actors. Please let len(epsilons) == no_envs*no_actors.")


        world_size = no_actors +2 #(+ Learner proces and Memory process)
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
            "device"                               :self.device,
            "transition_queue_from_memory"         :transition_queue_from_memory,
            "update_priorities_queue_to_memory"    :update_priorities_queue_to_memory,
            "con_actors"                           :con_learner_actor,
            "con_replay_memory"                    :con_learner_memory,
            "eval_freq"                            :eval_freq,
            "env"                                  :self.env,
            "env_config"                           :self.env_config,
            "tb_log_dir"                           :self.tb_log_dir,
            "update_tb"                            :self.update_tb
        }

        learner_process = Process(target=self._init_process, 
                                  args=(0, 
                                        world_size, 
                                        learner, 
                                        learner_args))
        learner_process.start()
        
        """
            Memory Process
        """
        mem_args = {
            "capacity"                          :self.replay_mem_size,
            "alpha"                             :self.alpha,
            "beta"                              :self.beta,
            "batch_size"                        :batch_size,
            "transition_queue_to_memory"        :transition_queue_to_memory,
            "transition_queue_from_memory"      :transition_queue_from_memory,
            "update_priorities_queue_to_memory" :update_priorities_queue_to_memory,
            "con_learner"                       :con_memory_learner,
            "replay_size_before_sampling"       :replay_size_before_sample if not (replay_size_before_sample is None) else min(batch_size, int(self.replay_memory*0.25)),
            "tb_log_dir"                        :self.tb_log_dir,
            "update_tb"                         :self.update_tb
            }
        
        print("Memory Process")
        memory_process = Process(target = self._init_process,
                                  args=(1, 
                                        world_size,
                                        experienceReplayBuffer,
                                        mem_args))

        memory_process.start()

        """
            Actor Processes
        """
        actor_args = { 
            "train_steps"                   :training_steps, 
            "max_actions_per_episode"       :max_actions_per_episode, 
            "update_policy"                 :policy_update, 
            "size_local_memory_buffer"      :size_local_memory_buffer, 
            "model"                         :self.policy_net,
            "model_config"                  :self.policy_config,
            "env"                           :self.env,
            "env_config"                    :self.env_config,
            "no_envs"                       :no_envs,
            "device"                        :self.device,
            "beta"                          :beta,
            "discount_factor"               :discount_factor,
            "transition_queue_to_memory"    :transition_queue_to_memory,
            "n_step"                        :n_step
            }

        split = 0
        for rank in range(no_actors):
            next_split = split + no_envs
            actor_args["epsilon"] = epsilons[split:next_split]
            actor_args["con_learner"] = con_actor_learner[rank] 
            
            split = next_split
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

        memory_process.join()
        learner_process.join()


    def _init_process(self, rank, size, fn, args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.2'
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, args)
        
    

        

