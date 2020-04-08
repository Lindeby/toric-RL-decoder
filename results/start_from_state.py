import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.nn.torch.NN import NN_11
import gym, gym_ToricCode, torch
import numpy as np
from src.util_actor import selectAction



if __name__ == "__main__":

    size = 5
    device = 'cpu'
    net_path = "network/converged/Size_5_NN_11_17_Mar_2020_22_33_59.pt"
    max_steps = 75
    start_error = np.loadtxt("data/checkpoints/5/cp_id0_size_5_p_0.05_failed_syndromes_9.txt")[0,:]
    start_error = np.array(start_error, dtype=np.int).reshape((2,size,size))


    env_config = {  "size": size,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
                }

    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }
    
    model = NN_11(model_config["system_size"], 3, device)
    model.load_state_dict(torch.load(net_path, map_location=device))
    model.eval()

    env = gym.make('toric-code-v0', config=env_config)
    env.reset()
    state = env.createSyndromOpt(start_error)
    env.qubit_matrix = start_error
    env.state = state

    env.plotToricCode(state, 'step_0')

    terminal_state = False
    no_steps = 0
    while not terminal_state and no_steps < max_steps:

        action, q_values = selectAction(number_of_actions=3,
                                                epsilon=0,
                                                grid_shift=int(size/2),
                                                toric_size=env.system_size,
                                                state=state,
                                                model=model,
                                                device=device)
        q_value = q_values[action[-1]-1]

        next_state, reward, terminal_state, _ = env.step(action)
        state = next_state
        no_steps += 1
        env.plotToricCode(state, 'step_{}'.format(no_steps))









