from src.ReplayMemory import PrioritizedReplayMemory
from queue import Empty
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
from src.Learner_mpi import learner

def io(memory_args, learner_args):
    
    memory_capacity = memory_args["capacity"]
    memory_alpha = memory_args["alpha"]
    memory_beta = memory_args["beta"]
    replay_size_before_sampling = memory_args["replay_size_before_sampling"]
    base_comm = memory_args["mpi_base_comm"]
    learner_rank = memory_args["mpi_learner_rank"] 
    batch_in_queue_limit = memory_args["batch_in_queue_limit"]
    batch_size = memory_args["batch_size"]
    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)

    
     
    ctx = mp.get_context('spawn')
    learner_io_queue = ctx.Queue()
    io_learner_queue = ctx.Queue() 
    con_io, con_learner = ctx.Pipe(duplex=True)
    
    learner_args["learner_io_queue"] = learner_io_queue
    learner_args["io_learner_queue"] = io_learner_queue
    learner_args["con_io"] = con_io
     
    learner_process = ctx.Process(target=learner, args=(learner_args,))
    learner_process.start()

    world_size = base_comm.Get_size()

    latest_network = None
    latest_network_id = None
    actors_current_network_id = None
    start_learning = False
    
    # Send initial network parameters to actors
    msg, network = learner_io_queue.get() #Blocks until there is a network in the queue to send
    if msg == "weights":
        latest_network = network
        print(network)
        latest_network_id = 0
        actors_current_network_id = 0
        msg = ("weights", network)
        base_comm.bcast(msg, root=learner_rank)
        
    while(True):
        
         # Send new weights if there are new avaliable
         if actors_current_network_id < latest_network_id:
             print("IO sending new weights")
             msg = ("weights", network)
             base_comm.bcast(msg, root=learner_rank)
             actors_current_network_id = latest_network_id
         else:
             msg = ("continue", None)
             base_comm.bcast(msg, root=learner_rank)

         # Gather transitions from actors
         actor_transitions = []
         actor_transitions = base_comm.gather(actor_transitions, root = learner_rank)

         # Insert transitions to replay memor
         for a in range(0, world_size):
             if a == learner_rank:
                 continue
             a_transitions = actor_transitions[a]
             
             for i in range(len(a_transitions)):
                 replay_memory.save(a_transitions[i][0], a_transitions[i][1])
         
         # Sample sample transitions utill there are x in queue to learner
         if replay_memory.filled_size() >= replay_size_before_sampling:
             start_learning = True
         
         while(start_learning and io_learner_queue.qsize() < batch_in_queue_limit):
             transitions, weights, indices = replay_memory.sample(batch_size, memory_beta)
             data = (transitions, weights, indices)
             io_learner_queue.put(data)
    
         # empty queue from learner 
         print("learner_io size: ",learner_io_queue.qsize())    
         while(learner_io_queue.empty == False):
         
             msg, item = learne_io_queue.get()
             if msg == "weights":
                 latest_network = item
                 latest_network_id +=1

             elif msg == "priorities":
                 # Update priorities
                 indices, priorities = item
                 replay_memory.priority_update(indices, priorities)
                 
             elif msg == "terminate":
                 print("IO received termination")
                 break
    
    msg = ("terminate", None)
    base_comm.bcast(msg, root=learner_rank) 
    # Empty all queues before termination and signal learner
    try:
        while True:
            learner_io_queue.get_nowait()
    except Empty:
        pass

    try:
        while True: 
            io_learner_queue.get_nowait()
    except Empty:
        pass
    
    learner_io_queue.close()
    io_learner_queue.close()
    print("IO done")
    con_learner.send("done")

    learner_process.join() 

   
