import sys
#sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')

from src.evaluation import evaluate
from src.nn.torch.NN import NN_11
from src.nn.torch.ResNet import ResNet18
import torch

from torch.utils.tensorboard import SummaryWriter
import numpy as np

if __name__ == "__main__":

    device = 'cuda'
    no_episodes = 20000 
    
    env_config = {  "size": 9,
                "min_qubit_errors": 0,
                "p_error": 0.1
                }

    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }
    #model = NN_11(model_config["system_size"], 3, device)
    model = ResNet18()
    model.load_state_dict(torch.load("runs/15_Apr_2020_20_06_29/Size_9_ResNet_15_Apr_2020_20_06_29.pt", map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    p_error = np.linspace(0.06, 0.2, 8, endpoint=True)
    for p in p_error:
        success_rate, ground_state_rate, average_number_of_steps_list, mean_q_list, failed_syndroms = evaluate( model,
                                                                                                                'toric-code-v0',
                                                                                                                env_config,
                                                                                                                int(env_config["size"]/2),
                                                                                                                device,
                                                                                                                [p],
                                                                                                                num_of_episodes=no_episodes,
                                                                                                                epsilon=0.0,
                                                                                                                num_of_steps=75,
                                                                                                                plot_one_episode=False, 
                                                                                                                minimum_nbr_of_qubit_errors=0)

        tb = SummaryWriter(log_dir='runs/test_size_{}_steps_{}'.format(env_config["size"], no_episodes))

        # for i, p in enumerate(p_error):
        tb.add_scalar("Performance/Ground State Rate", ground_state_rate[0], p*100)
        tb.add_scalar("Performance/Success Rate", success_rate[0], p*100)
        tb.add_scalar("Performance/Mean Q", mean_q_list[0], p*100)
        tb.add_scalar("Performance/Avg No Steps", average_number_of_steps_list[0], p*100)


        tb.close()
