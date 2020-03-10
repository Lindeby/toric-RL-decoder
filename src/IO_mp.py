from src.ReplayMemory import PrioritizedReplayMemory
from queue import Empty
import time
import multiprocessing as mp
from src.Learner_mpi import learner
from torch.utils.tensorboard import SummaryWriter


def io(memory_args):
    

    tb = SummaryWriter(log_dir="runs/" + memory_args["save_date"]+"_IO", filename_suffix="_IO")
    t0 = time.time()
    iteration_count = 0
    itr_time = 0

    memory_capacity = memory_args["capacity"]
    memory_alpha = memory_args["alpha"]
    memory_beta = memory_args["beta"]
    replay_size_before_sampling = memory_args["replay_size_before_sampling"]
    batch_in_queue_limit = memory_args["batch_in_queue_limit"]
    batch_size = memory_args["batch_size"]
    learner_io_queue = memory_args["learner_io_queue"]
    io_learner_queue = memory_args["io_learner_queue"]
    actor_io_queue = memory_args["actor_io_queue"]
    pipe_io_learner = memory_args["pipe_io_learner"]
    pipe_io_actor = memory_args["pipe_io_actor"]
    no_actors = memory_args["no_actors"] 


    replay_memory = PrioritizedReplayMemory(memory_capacity, memory_alpha)
      
    latest_network = None
    latest_network_id = None
    actors_current_network_id = None
    start_learning = False

    def send_to_actors(msg):
        print("IO: sending {} to actors.".format(msg[0]))
        for i in range(no_actors):
            pipe_io_actor[i].send(msg)
            pipe_io_actor[i].recv()
        print("IO: done.")
    
    # Send initial network parameters to actors
    msg, network = learner_io_queue.get() #Blocks until there is a network in the queue to send
    print("IO: received initial network weights from learner.")
    if msg == "weights":
        latest_network = network
        latest_network_id = 0
        actors_current_network_id = 0
        msg = ("weights", network)
        send_to_actors(msg) 

    while(True):
        s0 = time.time()
         # Send new weights if there are new avaliable
        if actors_current_network_id < latest_network_id:
            msg = ("weights", network)
            send_to_actors(msg)
            actors_current_network_id = latest_network_id
            tb.add_scalar('Time/Policy Network Update from IO to Actors', time.time()-t0)

         # empty queue of transtions from actors
        while(actor_io_queue.empty() == False):
            transitions = actor_io_queue.get()

            for i in range(len(transitions)):
                t,p = transitions[i]
                replay_memory.save(t, p)
            
         
         # Sample sample transitions utill there are x in queue to learner
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
            if msg == "weights":
                print("IO: received new weights from learner.")
                latest_network = item
                latest_network_id +=1

            elif msg == "priorities":
                 # Update priorities
                 indices, priorities = item
                 replay_memory.priority_update(indices, priorities)
                 
            elif msg == "prep_terminate":
                print("IO: received {} from learner.".format(msg))
                 
                msg = ("prep_terminate", None)

                for i in range(no_actors):
                    print("IO: sending {} to actor {}.".format(msg[0], i))
                    pipe_io_actor[i].send(msg)
                    pipe_io_actor[i].recv()
                    print("IO: recieved ACK from actor {}.".format(i)) 
                terminate = True
                break      
        if terminate:
            break  
        
        e0 = time.time()
        itr_time += e0 - s0
        iteration_count += 1
        if iteration_count >= 100:
            tb.add_scalar('Time/Avg_Itr_Time_IO', itr_time/iteration_count)
            iteration_count = 0
            itr_time = 0

    
    tb.close()
    print("IO: clearing any transitions from actors.")
    try:
        while True:
            actor_io_queue.get_nowait()
    except Empty:
        pass

    actor_io_queue.close() 
    print("IO: transition queue from actor clear and closed.")

    msg = ("terminate", None)
    print("IO: sending {} to actors.".format(msg[0]))
    for i in range(no_actors):
       pipe_io_actor[i].send(msg)
       pipe_io_actor[i].recv()
    print("IO: recieved ACK from actor {}.".format(i))
    try:
        while True:
            learner_io_queue.get_nowait()
    except Empty:
        pass

    msg = ("Ok", None)
    print("IO: sending ACK to Learner.")
    pipe_io_learner.send(msg)

    while True:
        msg = pipe_io_learner.recv()
        msg, _ = msg
        print(msg)
        if msg == "terminate":
            break
    
    msg = ("Ok", None)
    pipe_io_learner.send(msg)
    print("IO: terminated.")

   
