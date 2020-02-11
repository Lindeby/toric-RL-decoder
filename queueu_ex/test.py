import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process, Queue, Pipe
from queue import Empty
import time



def p(rank, size, args):
    if rank == 0:
        #master
        print("hello from master")
        
        q = args["queue"]
        con_memory = args["con_memory"]
        con_workers = args["con_workers"]
        
        received = 0
        while True:
            item = q.get()
            print(item)
            received +=1

            if received >= 100:
                print("master: prep terminate memory")
                msg = "prep_terminate"
                con_memory.send(msg)
                back = con_memory.recv()
                print(back)
                msg = "prep_terminate"
                for w in range(size-2):
                    print("prep_terminate worker ",w)
                    con_workers[w].send(msg)
                    back = con_workers[w].recv()
                    print(back)
                
                msg = "terminate"
                for w in range(size-2):
                    print("terminate worker ",w)
                    con_workers[w].send(msg)
                    back = con_workers[w].recv()
                    print(back)
                    
                msg = "terminate"
                print("terminate memory")
                con_memory.send(msg)
                back = con_memory.recv()
                print(back)
                
                try:
                    while True:
                        q.get_nowait()
                except Empty:
                    pass
                #while not q.empty():
                #    q.get()
                #q.close()   
                break    
                
            #time.sleep(1)
        print("master done")

    elif rank == 1:
        #memory
        print("hello from memmory")
        con_master = args["con_master"]
        q_to_mem = args["queue_to_mem"]
        q_from_mem = args["queue_from_mem"]
        
        while True:
            if con_master.poll():
                msg = con_master.recv()
                if msg == "prep_terminate":
                    
                    con_master.send("ok")
                    break
            if not q_to_mem.empty():
                item = q_to_mem.get()
                q_from_mem.put(item)
        
        while True:
            msg = con_master.recv()
            if msg == "terminate":
                 
                try:
                    while True:
                        q_to_mem.get_nowait()
                except Empty:
                    pass
                #while not q_to_mem.empty():
                #    q_to_mem.get()
                q_to_mem.close()
                q_from_mem.close()
                con_master.send("ok")
                break
        print("memory done")

    else:
        #worker
        print("hello from worker",rank)
        i = 0
        q = args["queue_to_mem"]
        con_master = args["con_master"]
        while True:
            
            if con_master.poll():
                msg = con_master.recv()
                if msg == "prep_terminate":
                    q.close()
                    con_master.send("ok")
                    break;
            
            item = 'worker '+str(rank) + ': '+str(i)
            q.put(item)
            i+=1
        while True:
            msg = con_master.recv()
            if msg == "terminate":
                con_master.send("ok")
                break

        print("worker ", rank," done")


def init_process(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


num_workers = 1
world_size = num_workers +2

q_to_mem = Queue()
q_from_mem = Queue()

con_master_mem_1, con_master_mem_2 = Pipe(duplex = True)

con_recv_worker = []
con_send_worker = []
for i in range(num_workers):
    con_1, con_2 = Pipe(duplex = True)
    con_recv_worker.append(con_1)
    con_send_worker.append(con_2)

#args = {}
args = {"queue":q_from_mem,
        "con_memory":con_master_mem_1,
        "con_workers":con_send_worker
        }

master = Process(target = init_process,
                 args = (0, world_size, p, args))

master.start()
print("starting maste")

args = {"queue_from_mem":q_from_mem,
        "queue_to_mem":q_to_mem,
        "con_master":con_master_mem_2}

memory = Process(target = init_process,
                 args = (1, world_size, p, args))

memory.start()
print("starting memory")
args = {"queue_to_mem":q_to_mem}


workers = []
for w in range(num_workers): 
    args["con_master"] = con_recv_worker[w]
    worker = Process(target = init_process,
                     args = (w+2, world_size, p, args))
    workers.append(worker)
    worker.start()
    print("starting worker ",w)


for w in range(num_workers):
    workers[w].join()
    print("worker ",w," joined")        
memory.join()
print("memory joined")
master.join()
print("master joined")

