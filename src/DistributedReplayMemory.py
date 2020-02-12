
from .ReplayMemory import PrioritizedReplayMemory

from queue import Empty


def experienceReplayBuffer(rank, world_size, args):
    """ 
        args = {"capacity",
                "alpha",
                "beta",
                "batch_size",
                "transition_queue_to_memory",
                "transition_queue_form_memory",
                "update_priorities_queue_to_memory",
                "con_learner"
                }
    """
   
    transition_queue_to_memory = args["transition_queue_to_memory"]
    transition_queue_from_memory = args["transition_queue_from_memory"]
    update_priorities_queue_to_memory = args["update_priorities_queue_to_memory"]
    capacity = args["capacity"]
    alpha = args["alpha"]
    beta = args["beta"]
    batch_size = args["batch_size"]
    con_learner = args["con_learner"]
    memory = PrioritizedReplayMemory(capacity, alpha)
    size_before_sample = args["replay_size_before_sampling"]
    items_in_mem = 0

    while(True):
        
        if con_learner.poll():
            msg = con_learner.recv()
            if msg == "prep_terminate":
                con_learner.send("ok")
                break

        #Receive transitions from actors
        for _ in range(100):
            if transition_queue_to_memory.empty():
                break
            
            #transition, priority = transition_queue_to_memory.get()
            back = transition_queue_to_memory.get()

            #state, action, reward, next_state, terminal, priority = zip(*back)
            transition, priority = zip(*back)
            items_in_mem += len(transition)

            for i in range(len(back)):
                memory.save(transition[i], priority[i])
        
        #Sample batch of transitions to learner

        if items_in_mem > size_before_sample:
    
            while transition_queue_from_memory.qsize() < 5:
                transition, weights, indices = memory.sample(batch_size, beta)
                transition_queue_from_memory.put((transition, weights, indices))



        for _ in range(10):
            if update_priorities_queue_to_memory.empty():
                break
            
            update = update_priorities_queue_to_memory.get()
            priorities, indices = zip(*update)
            memory.priority_update(indices, priorities)
    
    while True:
        # Ready to terminate nothing more should be sent to learner
        
        # Still empty queue since actors might not have recived 
        # instructions to terminate
        if con_learner.poll():
            msg = con_learner.recv()
            if msg == "terminate":
                # Empty queues to memory befor termination
                try:
                    while True:
                        transition_queue_to_memory.get_nowait()
                except Empty:
                    pass
                
                try: 
                    while True:
                        update_priorities_queue_to_memory.get_nowait()
                except Empty:
                    pass

                try:
                    while True:
                        transition_queue_from_memory.get_nowait()
                except Empty:
                    pass


                transition_queue_from_memory.close()
                transition_queue_to_memory.close()
                update_priorities_queue_to_memory.close()
                con_learner.send("ok")
                break
