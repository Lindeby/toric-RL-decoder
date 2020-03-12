import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.evaluation import evaluate
from src.nn.torch.NN import NN_11
import torch

if __name__ == "__main__":
    
    env_config = {  "size": 3,
                "min_qubit_errors": 0,
                "p_error": 0.25
                }

    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }
    model = NN_11(model_config["system_size"], 3, 'cpu')
    model.load_state_dict(torch.load("network/Size_3_NN_11_11_Mar_2020_22_13_15.pt", map_location='cpu'))
    model.eval()


    epsilon = [0.1]
    evaluate(model, 'toric-code-v0', env_config, int(env_config["size"]/2), 'cpu', epsilon, num_of_episodes=1,
    num_actions=3, epsilon=0.0, num_of_steps=50, plot_one_episode=True, 
    show_network=False, minimum_nbr_of_qubit_errors=0, 
    print_Q_values=False)
