from src.ReplayMemory import PrioritizedReplayMemory
import time

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
    

    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)

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
            
         
         # Sample sample transitions until there are x in queue to learner
        if (not start_learning) and replay_memory.filled_size() >= replay_size_before_sampling:
             start_learning = True
         
        while(start_learning and io_learner_queue.qsize() < batch_in_queue_limit):
            transitions, weights, indices = replay_memory.sample(batch_size, memory_beta)
            data = (transitions, weights, indices)
            io_learner_queue.put(data)
    
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
