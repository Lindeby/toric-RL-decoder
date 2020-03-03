from src.actor import selectAction, generateTransition
from src.nn.torch.ResNet import ResNet18
import gym, gym_ToricCode

# saving
from pathlib import Path
from numpy import save
import time

def worker(T):
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

        mem_trans.append(transition)
        mem_q_v.append(qs)

        if terminal or steps_count > max_steps_per_ep:
            state = env.reset()
            steps_count = 0

        state = next_state
        steps_count += 1
        
    return mem_trans, mem_q_v


if __name__ == "__main__":
    num_transitions = 10
    start_time = time.time()
    mem_trans, mem_qs = worker(num_transitions)
    end_time = time.time()
    elapsed_time = end_time -start_time
    print("created ",num_transitions,"transitions in: ",elapsed_time)
    save_name = 'output_speed_test/transitions_'+str(0)+'.npy'
    save_name_q = 'output_speed_test/q_values_'+str(0)+'.npy'
    Path("output_speed_test").mkdir(parents=True, exist_ok=True)
    save(save_name, mem_trans)
    save(save_name_q, mem_qs)


