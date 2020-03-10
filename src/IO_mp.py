from src.ReplayMemory import PrioritizedReplayMemory
from queue import Empty
import time
import multiprocessing as mp
from src.Learner_mpi import learner

def io(memory_args):
    
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
        for i in range(no_actors):
            pipe_io_actor[i].send(msg)
            print("send to actors")
            pipe_io_actor[i].recv()
    
    # Send initial network parameters to actors
    msg, network = learner_io_queue.get() #Blocks until there is a network in the queue to send
    if msg == "weights":
        latest_network = network
        latest_network_id = 0
        actors_current_network_id = 0
        msg = ("weights", network)
        send_to_actors(msg) 

    while(True):
        
         # Send new weights if there are new avaliable
         if actors_current_network_id < latest_network_id:
             msg = ("weights", network)
             send_to_actors(msg)
             actors_current_network_id = latest_network_id
             print("IO sync")
             

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
                 latest_network = item
                 latest_network_id +=1

             elif msg == "priorities":
                 # Update priorities
                 indices, priorities = item
                 replay_memory.priority_update(indices, priorities)
                 
             elif msg == "prep_terminate":
                 print("IO  received termination")
                 
                 msg = ("prep_terminate", None)

                 for i in range(no_actors):
                     pipe_io_actor[i].send(msg)
                     
                     pipe_io_actor[i].recv()
                     print("recieved from ",i)
                              
                 terminate = True
                 break      
         if terminate:
             break  

    try:
        while True:
            actor_io_queue.get_nowait()
    except Empty:
        pass

    actor_io_queue.close() 

    msg = ("terminate", None)
    print("send terminate to actors")
    for i in range(no_actors):
       pipe_io_actor[i].send(msg)
       pipe_io_actor[i].recv()
    print("received ok from actors")
    try:
        while True:
            learner_io_queue.get_nowait()
    except Empty:
        pass

    msg = ("Ok", None)
    pipe_io_learner.send(msg)

    while True:
        msg = pipe_io_learner.recv()
        msg, _ = msg
        print(msg)
        if msg == "terminate":
            break
    
    msg = ("Ok", None)
    pipe_io_learner.send(msg)

   
