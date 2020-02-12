
from .ReplayMemory import PrioritizedReplayMemory
from torch.utils.tensorboard import SummaryWriter
from .ReplayMemory import PrioritizedReplayMemory
from queue import Empty


def experienceReplayBuffer(rank, world_size, args):
    """ 

    Params
    ======
    rank:           (int)
    world_size:     (int)
    args:           (dict)
    {   
        capacity                                (int)
        , alpha                                 (float)
        , beta                                  (float)
        , batch_size                            (int)
        , transition_queue_to_memory            (torch.multiprocessing.Queue)
        , transition_queue_from_memory          (torch.multiprocessing.Queue)
        , update_priorities_queue_to_memory     (torch.multiprocessing.Queue)
        , con_learner                           (torch.multuprocessing.Queue)
        , update_tb                             (int) frequensy to update tensorboard
        , tb_log_dir                            (String) tensorboard log dir
        
    }
    """
    
    # tensorboard
    tb = SummaryWriter(log_dir=args["tb_log_dir"]+"_memory", filename_suffix="_memory")
    update_tb = args["update_tb"]


    trans_q_from_mem_avg_size = 0
    trans_q_to_mem_avg_size = 0
    update_prio_q_to_memory_avg_size = 0
    timestep = 0

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
            
            back = transition_queue_to_memory.get()

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
        
        # data to eventually log
        trans_q_from_mem_avg_size           += transition_queue_from_memory.qsize()
        trans_q_to_mem_avg_size             += transition_queue_to_memory.qsize()
        update_prio_q_to_memory_avg_size    += update_priorities_queue_to_memory.qsize()
        timestep += 1

        # write to tensorboard
        u = timestep % update_tb 
        if u == 0:
            data = [trans_q_from_mem_avg_size / update_tb
                    , trans_q_to_mem_avg_size / update_tb         
                    , update_prio_q_to_memory_avg_size / update_tb
                    ]
            writeToTB(tb, data, timestep, update_tb)
            trans_q_from_mem_avg_size           = 0
            trans_q_to_mem_avg_size             = 0
            update_prio_q_to_memory_avg_size    = 0

    tb.close()
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
                
 
def writeToTB(tb, data, timestep, update):
    tb.add_scalar("Queue/Avg_Over_{}_TransQ_Size(fromMemory)".format(update), data[0], timestep)
    tb.add_scalar("Queue/Avg_Over_{}_TransQ_Size(toMemory)".format(update), data[1], timestep)
    tb.add_scalar("Queue/Avg_Over_{}_PrioQ_Size".format(update), data[2], timestep)

    