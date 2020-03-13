from src.ReplayMemory import PrioritizedReplayMemory
import numpy as np
import time
from datetime import datetime

def io(memory_args):
    
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
    log_priority_dist = memory_args["log_priority_dist"]
    if log_priority_dist:
        log_write_frequency                 = memory_args["log_write_frequency"]
        log_priority_sample_max             = memory_args["log_priority_sample_max"]
        log_priority_sample_interval_size   = memory_args["log_priority_sample_interval_size"]
        samples_actor   = np.zeros(int(log_priority_sample_max/log_priority_sample_interval_size))
        samples_learner = np.zeros(int(log_priority_sample_max/log_priority_sample_interval_size))
        start_time = datetime.now().strftime("%d_%b_%Y-%H:%M:%S")
        actor_path = "data/sample_distribution_actor_" + start_time + ".data"
        learner_path = "data/sample_distribution_learner_" + start_time + ".data"

        header = "HEADER:::::min={}, max={}, interval={}".format(0, log_priority_sample_max, log_priority_sample_interval_size)

        # write info in header
        appendToFile(header, actor_path  , timestamp=False)
        appendToFile(header, learner_path, timestamp=False)




    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)

    log_count_actor   = 0
    log_count_learner = 0
    start_learning = False
    total_amout_transitions = 0
    while(True):

        # empty queue of transtions from actors
        while(actor_io_queue.empty() == False):
            
            transitions = actor_io_queue.get()
            for i in range(len(transitions)):
                t,p = transitions[i]
                replay_memory.save(t, p)
                total_amout_transitions +=1

                
                # log distribution
                if log_priority_dist:
                    samples_actor[min(int(p/log_priority_sample_interval_size), len(samples_actor)-1)] += 1
            
            # append logged priorities from actor to file
            log_count_actor += 1
            if log_priority_dist and log_count_actor >= log_write_frequency:
                log_count_actor = 0
                appendToFile(samples_actor, actor_path)
                samples_actor = np.zeros(int(log_priority_sample_max/log_priority_sample_interval_size))

            
         # Sample sample transitions until there are x in queue to learner
        if (not start_learning) and replay_memory.filled_size() >= replay_size_before_sampling:
             start_learning = True
         
        while(start_learning and io_learner_queue.qsize() < batch_in_queue_limit):
            transitions, weights, indices, priorities = replay_memory.sample(batch_size, memory_beta)
            data = (transitions, weights, indices)
            io_learner_queue.put(data)

            # log distribution
            if log_priority_dist:
                samples_learner[np.minimum((np.array(priorities)/log_priority_sample_interval_size).astype(np.int), len(samples_actor)-1)] += 1


            # append logger priorities going to actor to file
            log_count_learner += 1
            if log_priority_dist and log_count_learner >= log_write_frequency:
                log_count_learner = 0 
                appendToFile(samples_learner, learner_path)
                samples_learner = np.zeros(int(log_priority_sample_max/log_priority_sample_interval_size))

        # empty queue from learner
        terminate = False
        while(not learner_io_queue.empty()):
         
            msg, item = learner_io_queue.get()
             
            if msg == "priorities":
                # Update priorities
                indices, priorities = item
                replay_memory.priority_update(indices, priorities)            
            elif msg == "terminate":
                print("Totel amount of generated transitions: ",total_amout_transitions)



def appendToFile(data, path, timestamp=True):
    dt = datetime.now().strftime("%H:%M:%S") 
    with open(path, 'a') as f:
        f.write(dt + ":::::" + " ".join(map(str, data)) + "\n")
