from src.ReplayMemory import PrioritizedReplayMemory
import numpy as np
import time
import nvgpu
from datetime import datetime

could_import_tb=True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    could_import_tb=False
    print("Could not import tensorboard. No logging will occur.")


def io(memory_args, actor_args, learner_args):
    
    memory_capacity             = memory_args["capacity"]
    memory_alpha                = memory_args["alpha"]
    memory_beta                 = memory_args["beta"]
    replay_size_before_sampling = memory_args["replay_size_before_sampling"]
    batch_in_queue_limit        = memory_args["batch_in_queue_limit"]
    batch_size                  = memory_args["batch_size"]
    learner_io_queue            = memory_args["learner_io_queue"]
    io_learner_queue            = memory_args["io_learner_queue"]
    actor_io_queue              = memory_args["actor_io_queue"]
    
    # Logging of priority distributions
    should_log = memory_args["log_priority_dist"]
    if should_log and could_import_tb:
        start_time                          = memory_args["start_time"]
        tb_write_dir                        = "runs/{}/IO/".format(start_time)
        tb_write_frequency                  = memory_args["log_write_frequency"]
        tb_priority_sample_max              = memory_args["log_priority_sample_max"]
        tb_priority_sample_interval_size    = memory_args["log_priority_sample_interval_size"]
        samples_actor   = np.zeros(int(tb_priority_sample_max/tb_priority_sample_interval_size))
        samples_learner = np.zeros(int(tb_priority_sample_max/tb_priority_sample_interval_size))

        tb = SummaryWriter(tb_write_dir)
        tb_nvidia_log_freq = 10 #seconds

    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)

    if should_log:
        
        tb_setup_string = ("env_size: {}  \n"
                           "learning_rate: {}  \n"
                           "learner_update_policy: {}  \n"
                           "learner_optimizer: {}  \n"
                           "learner_device: {}  \n"
                           "learner_job_max_time: {}  \n"
                           "learner_save_date: {}  \n"
                           "learner_eval_no_episodes: {}  \n"
                           "learner_eval_freq: {}  \n"
                           "actor_max_actions_per_episode: {}  \n"
                           "actor_size_local_memory_buffer: {}  \n"
                           "actor_no_envs: {}  \n"
                           "no_cuda_actors: {}  \n"
                           "no_cpu_actors: {}  \n"
                           "env_p_error_interval_start: {}  \n"   
                           "env_p_error_interval_final: {}  \n"   
                           "env_p_error_interval_increase: {}  \n"
                           "env_p_error_strategy: {}  \n"
                           "replay_memory_size: {}  \n"                            
                           "replay_memory_alpha: {}  \n"                 
                           "replay_memory_beta: {}  \n"                  
                           "replay_memory_size_before_sampeling: {}  \n" 
                           "replay_memory_batch_in_queue_limit: {}  \n"  
                           "log_priority_dist: {}  \n"                   
                           "log_write_frequency: {}  \n"                 
                           "log_priority_sample_max: {}  \n"             
                           "log_priority_sample_interval_size: {}  \n"
                           "batch_size: {}  \n"
                           "discount_factor: {}  \n").format(learner_args["env_config"]["size"],
                                                             learner_args["learning_rate"],
                                                             learner_args["policy_update"],
                                                             learner_args["optimizer"],
                                                             learner_args["device"],
                                                             learner_args["job_max_time"],
                                                             learner_args["save_date"],
                                                             learner_args["learner_eval_no_episodes"],
                                                             learner_args["learner_eval_freq"],
                                                             actor_args["max_actions_per_episode"],
                                                             actor_args["size_local_memory_buffer"],
                                                             actor_args["no_envs"],
                                                             actor_args["no_cuda_actors"],
                                                             actor_args["no_cpu_actors"],
                                                             actor_args["env_p_error_start"],
                                                             actor_args["env_p_error_final"],
                                                             actor_args["env_p_error_delta"],
                                                             actor_args["env_p_error_strategy"],
                                                             memory_args["capacity"],
                                                             memory_args["alpha"],
                                                             memory_args["beta"],
                                                             memory_args["replay_size_before_sampling"],
                                                             memory_args["batch_in_queue_limit"],
                                                             memory_args["log_priority_dist"],
                                                             memory_args["log_write_frequency",
                                                             memory_args["log_priority_sample_max"],
                                                             memory_args["log_priority_sample_interval_size"],
                                                             memory_args["batch_size"],
                                                             actor_args["discount_factor"])

        print(tb_setup_string)


    log_count_actor   = 0     
    log_count_learner = 0
    count_gen_trans   = 0
    count_cons_trans  = 0
    count_total_gen_trans = 0
    count_total_cons_trans = 0
    start_learning = False
    total_amout_transitions = 0
    nvidia_log_time = time.time()
    stop_watch = time.time()
    while(True):

        # empty queue of transtions from actors
        while(actor_io_queue.empty() == False):
            
            transitions = actor_io_queue.get()
            for i in range(len(transitions)):
                t,p = transitions[i]
                replay_memory.save(t, p)
                total_amout_transitions +=1


                # log distribution
                if should_log:
                    count_gen_trans += 1
                    samples_actor[min(int(p/tb_priority_sample_interval_size), len(samples_actor)-1)] += 1
            
            # append logged priorities from actor to file
            log_count_actor += 1
            if should_log and could_import_tb and log_count_actor >= tb_write_frequency:
                log_count_actor  = 0
                count_total_gen_trans  += count_gen_trans
                count_total_cons_trans += count_cons_trans
                t = time.time()
                tb.add_histogram("Distribution/Actor Distribution", samples_actor)
                tb.add_scalars("Data/", {"Total Consumption":count_total_cons_trans, "Total Generation":count_total_gen_trans})
                tb.add_scalars("Data/", {"Consumption per Second":count_cons_trans/(t-stop_watch), "Generation per Second":count_gen_trans/(t-stop_watch)})
                stop_watch = time.time()
                count_gen_trans  = 0
                count_cons_trans = 0
                samples_actor    = np.zeros(int(tb_priority_sample_max/tb_priority_sample_interval_size))

            if should_log and nvidia_log_time + tb_nvidia_log_freq > time.time():
                nvidia_log_time = time.time() 
                gpu_info = nvgpu.gpu_info()
                
                for i in gpu_info:
            
                    gpu = '{} {}'.format(i['type'], i['index'])
                    mem_total = i['mem_total']
                    mem_used = i['mem_used'] 
                    tb.add_scalars(gpu, {'mem_total':mem_total,
                                           'mem_used':mem_used})


            
         # Sample sample transitions until there are x in queue to learner
        if (not start_learning) and replay_memory.filled_size() >= replay_size_before_sampling:
             start_learning = True
         
        while(start_learning and io_learner_queue.qsize() < batch_in_queue_limit):
            transitions, weights, indices, priorities = replay_memory.sample(batch_size, memory_beta)
            data = (transitions, weights, indices)
            io_learner_queue.put(data)

            # log distribution
            if should_log:
                count_cons_trans += batch_size
                samples_learner[np.minimum((np.array(priorities)/tb_priority_sample_interval_size).astype(np.int), len(samples_actor)-1)] += 1


            # append logger priorities going to actor to file
            log_count_learner += 1
            if should_log and could_import_tb and log_count_learner >= tb_write_frequency:
                log_count_learner = 0 
                tb.add_histogram("Distribution/Learner Distribution", samples_actor)
                samples_learner = np.zeros(int(tb_priority_sample_max/tb_priority_sample_interval_size))

        # empty queue from learner
        terminate = False
        while(not learner_io_queue.empty()):
         
            msg, item = learner_io_queue.get()
             
            if msg == "priorities":
                # Update priorities
                indices, priorities = item
                replay_memory.priority_update(indices, priorities)            
            elif msg == "terminate":
                if should_log and could_import_tb:
                    tb.close()
                print("Total amount of generated transitions: ",total_amout_transitions)
