import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.evaluation import evaluate
from src.nn.torch.NN import NN_11
import torch

from torch.utils.tensorboard import SummaryWriter
import numpy as np

if __name__ == "__main__":
    
    env_config = {  "size": 3,
                "min_qubit_errors": 0,
                "p_error": 0.1
                }

    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }
    model = NN_11(model_config["system_size"], 3, 'cpu')
    model.load_state_dict(torch.load("network/latest/Size_3_NN_11_14_Mar_2020_04_39_05.pt", map_location='cpu'))
    model.eval()


    p_error = np.linspace(0.1, 0.3, 20, endpoint=True)
    success_rate, ground_state_rate, average_number_of_steps_list, mean_q_list, failed_syndroms = evaluate( model,
                                                                                                            'toric-code-v0',
                                                                                                            env_config,
                                                                                                            int(env_config["size"]/2),
                                                                                                            'cpu',
                                                                                                            p_error,
                                                                                                            num_of_episodes=1,
                                                                                                            epsilon=0.0,
                                                                                                            num_of_steps=50,
                                                                                                            plot_one_episode=True, 
                                                                                                            minimum_nbr_of_qubit_errors=0)

    tb = SummaryWriter(log_dir='runs/test/',filename_suffix='_test')
    
    for i, p in enumerate(p_error):
        tb.add_scalar("Testing/Evaluation of Network", ground_state_rate[i], p*100)

    tb.close()