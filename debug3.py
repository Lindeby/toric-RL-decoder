from src.ToricCode import ToricCode
from src.actor import selectAction, generateTransition
from src.nn.torch.ResNet import ResNet18
import gym, gym_ToricCode
from torch.multiprocessing import Process

# saving
from pathlib import Path
from numpy import save

# debug
import objgraph, sys, psutil, gc, torch


def worker(T):
    vm1 = psutil.virtual_memory()

    config =    {  "size": 9,
                "min_qubit_errors": 0,
                "p_error": 0.5
                }
                 
    model = ResNet18()

    env = gym.make('toric-code-v0', config=config)
    state = env.reset()

    mem_trans = []
    mem_q_v   = []
    max_steps_per_ep = 5
    steps_count = 0

    for t in range(T):

        a, qs = selectAction(
            number_of_actions = 3,
            epsilon           = 0.4,
            grid_shift        = int(env.system_size/2),
            toric_size        = env.system_size,
            state             = state,
            model             = model,
            device            = 'cpu'
        )

        next_state, reward, terminal, _ = env.step(a)

        transition = generateTransition(a, reward, int(env.system_size/2), state, next_state, terminal)

        # mem_trans.append(transition)
        # mem_q_v.append(qs)

        if terminal or steps_count > max_steps_per_ep:
            state = env.reset()
            steps_count = 0
            vm2 = psutil.virtual_memory()
            print("Total RAM use: {}mb".format(vm2.active/1048576))
            print("RAM increase: {}mb".format((vm2.active - vm1.active)/1048576))
            vm1 = vm2


        steps_count += 1
        
    # save_name = 'output_speed_test/transitions_'+str(0)+'.npy'
    # save_name_q = 'output_speed_test/q_values_'+str(0)+'.npy'
    # Path("output_speed_test").mkdir(parents=True, exist_ok=True)
    # save(save_name, mem_trans)
    # save(save_name_q, mem_q_v)


if __name__ == "__main__":
    objgraph.show_growth(limit=5)

    p = Process(target=worker, args=[10])
    p.start()
    p.join()

    objgraph.show_growth()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass





