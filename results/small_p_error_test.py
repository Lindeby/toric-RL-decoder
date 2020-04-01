import sys
sys.path.append('/home/adam/Documents/school/thesis/toric-RL-decoder')
from src.util import incrementalMean
from src.util_actor import selectAction
import gym, gym_ToricCode
import numpy as np


def generateRandomError(matrix, p_error):
    qubits = np.random.uniform(0, 1, size=(2, matrix.shape[1], matrix.shape[2]))
    error = qubits > p_error
    no_error = qubits < p_error
    qubits[error] = 0
    qubits[no_error] = 1
    pauli_error = np.random.randint(3, size=(2, matrix.shape[1], matrix.shape[2])) + 1
    matrix = np.multiply(qubits, pauli_error)

    return matrix.astype(np.int)


def generateNRandomErrors(matrix, n):
    errors = np.random.randint(3, size = n) + 1
    qubit_matrix_error = np.zeros(2*matrix.shape[1]**2)
    qubit_matrix_error[:n] = errors
    np.random.shuffle(qubit_matrix_error)
    matrix[:,:,:] = qubit_matrix_error.reshape(2, matrix.shape[1], matrix.shape[2])
    return matrix


def generateNPlusQRandomErrors(q, p_error, qubit_matrix):
    qubit_matrix = generateNRandomErrors(qubit_matrix, q)
    error_coords = np.nonzero(qubit_matrix)

    qubit_matrix2 = np.zeros(qubit_matrix.shape)
    qubit_matrix2 = generateRandomError(qubit_matrix2, p_error)

    qubit_matrix2[error_coords] = 0

    return qubit_matrix + qubit_matrix2



def prediction_smart(model, env, env_config, grid_shift, device, prediction_list_p_error, num_of_episodes=1, epsilon=0.0, num_of_steps=50, plot_one_episode=False, 
        show_network=False, show_plot=False, nbr_of_qubit_errors=0, print_Q_values=False):

        def comb(n, k):
            """
            A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
            """
            if 0 <= k <= n:
                ntok = 1
                ktok = 1
                for t in range(1, min(k, n - k) + 1):
                    ntok *= n
                    ktok *= t
                    n -= 1
                return ntok // ktok
            else:
                return 0

        model.to(device)
        model.eval()

        # init matrices 
        max_number_of_errors = 17
        ground_state_list               = np.zeros(len(prediction_list_p_error))
        error_corrected_list            = np.zeros(len(prediction_list_p_error))
        average_number_of_steps_list    = np.zeros(len(prediction_list_p_error))
        mean_q_list                     = np.zeros(len(prediction_list_p_error))
        P_l_list                        = np.zeros(len(prediction_list_p_error))

        size = env_config["size"]

        cfg = {"size":env_config["size"], "min_qubit_errors":env_config["min_qubit_errors"], "p_error":prediction_list_p_error[0]}
        env = gym.make(env, config=cfg)

        number_of_failed_syndroms_list      = np.zeros((3,max_number_of_errors))
        number_of_failed_syndroms_list[0,:] = np.linspace(0, max_number_of_errors-1, num=max_number_of_errors)
        
        # loop through different p_error
        for i, p_error in enumerate(prediction_list_p_error):
            ground_state            = np.ones(num_of_episodes,  dtype=bool)
            error_corrected         = np.zeros(num_of_episodes, dtype=bool)
            mean_steps_per_p_error  = 0
            mean_q_per_p_error      = 0
            steps_counter           = 0

            for j in range(num_of_episodes):
                print("p_error: {}, episode: {}".format(p_error, j))
                num_of_steps_per_episode = 0
                prev_action              = 0
                terminal_state           = 0

                state = env.reset(p_error=p_error)
                terminal_state = True
                # custom reset function
                while terminal_state:
                    qubit_matrix = np.zeros(state.shape, dtype=np.int)                                               # create new qubit_matrix
                    qubit_matrix = generateNPlusQRandomErrors(nbr_of_qubit_errors, p_error, qubit_matrix)   # apply new errors
                    state = env.createSyndromOpt(qubit_matrix)                                              # create the new syndrome
                    terminal_state = env.isTerminalState(state)

                # overwrite the envs data with our custome generated errors
                env.qubit_matrix = qubit_matrix
                env.state = state

                env.plotToricCode(state, "testing")

                number_of_qubit_flips = np.sum((env.qubit_matrix != 0))

                # plot initial state
                if plot_one_episode == True and j == 0 and i == 0:
                    env.plotToricCode(state, 'initial_syndrom')
                
                
                # solve syndrome                
                while not terminal_state and num_of_steps_per_episode < num_of_steps:
                    steps_counter += 1
                    num_of_steps_per_episode += 1

                    # choose greedy action
                    action, q_values = selectAction(number_of_actions=3,
                                                epsilon=epsilon,
                                                grid_shift=grid_shift,
                                                toric_size=env.system_size,
                                                state=state,
                                                model=model,
                                                device=device)
                    q_value = q_values[action[-1]-1]
                                        
                    next_state, reward, terminal_state, _ = env.step(action)

                    state = next_state
                    mean_q_per_p_error = incrementalMean(q_value, mean_q_per_p_error, steps_counter)
                    
                    if plot_one_episode == True and j == 0 and i == 0:
                        env.plotToricCode(state, 'step_'+str(num_of_steps_per_episode))

                mean_steps_per_p_error = incrementalMean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
                # save error corrected 
                error_corrected[j] = terminal_state # 1: error corrected # 0: error not corrected       
                
                # update groundstate
                ground_state[j] = env.evalGroundState()

                # count failed runs 
                if ground_state[j] == False:
                    number_of_failed_syndroms_list[2, number_of_qubit_flips] += 1
                elif ground_state[j] == True:
                    number_of_failed_syndroms_list[1, number_of_qubit_flips] += 1

            n_fail = np.zeros(max_number_of_errors)
            for k, item in enumerate(number_of_failed_syndroms_list[2,:]):
                if k < nbr_of_qubit_errors:
                    n_fail[k] = 0
                else:
                    n_fail[k] = item / comb(k, nbr_of_qubit_errors) 

            N_fail  = np.sum(n_fail)
            q       = nbr_of_qubit_errors
            n       = 2 * size**2
            p_conf  = p_error**q * (1-p_error)**(n-q)
            p_q     = comb(n,q) * p_conf 
            P_l     = p_q * N_fail / num_of_episodes

            success_rate                    = (num_of_episodes - np.sum(error_corrected)) / num_of_episodes
            error_corrected_list[i]         = success_rate            
            ground_state_change             = (num_of_episodes - np.sum(ground_state)) / num_of_episodes
            ground_state_list[i]            =  1 - ground_state_change
            average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
            mean_q_list[i]                  = np.round(mean_q_per_p_error, 3)
            P_l_list[i]                     = P_l

        return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, number_of_failed_syndroms_list, N_fail, P_l_list
   

from src.nn.torch.NN import NN_11
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == "__main__":
    no_episodes = 2
    p_error = [5e-2, 5e-3, 5e-4, 5e-5]
    device = 'cuda'
    size = 7

    env_config = {  "size": size,
                    "min_qubit_errors": 0,
                    "p_error": 0.1
                }

    model_config = {"system_size": env_config["size"],
                    "number_of_actions": env_config["size"]
                    }
    
    model = NN_11(model_config["system_size"], 3, device)
    model.load_state_dict(torch.load("network/mp/Size_7_NN_11_random_18_Mar_2020_18_17_52.pt", map_location='cpu'))
    model.eval()


    error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, number_of_failed_syndroms_list, n_fail, P_l = prediction_smart(model=model,
                    env='toric-code-v0',
                    env_config=env_config, 
                    grid_shift=int(env_config["size"]/2), 
                    device=device, 
                    prediction_list_p_error=p_error, 
                    num_of_episodes=no_episodes, 
                    epsilon=0.0, 
                    num_of_steps=75, 
                    plot_one_episode=False, 
                    show_network=False,
                    show_plot=False,
                    nbr_of_qubit_errors=int(env_config["size"]/2)+1,
                    print_Q_values=False)

    if size == 5:
        ground_state_conserved_theory = 0.9995275888133031 # see combinatorics file
        ground_state_failed_theory = 0.000472411186696901
    elif size == 7:
        ground_state_conserved_theory = 0.9999936414812806 # see combinatorics file
        ground_state_failed_theory = 6.35851871947911e-06
    elif size == 9:
        ground_state_conserved_theory = 0.9999999273112429 # see combinatorics file
        ground_state_failed_theory = 7.268875712609422e-08

    failure_rate = 1 - np.array(ground_state_list)
    asymptotic_fail = (failure_rate-ground_state_failed_theory)/ground_state_failed_theory * 100
    asymptotic_success = (np.array(ground_state_list)-ground_state_conserved_theory)/ground_state_conserved_theory * 100


    tb = SummaryWriter(log_dir='runs/evaluation/small_p_size_{}'.format(env_config["size"]))
    
    for i, p in enumerate(p_error):
        tb.add_scalar("Small p/Ground State Rate", ground_state_list[i], p)
        tb.add_scalar("Small p/Success Rate", error_corrected_list[i], p)
        tb.add_scalar("Small p/Mean Q", mean_q_list[i], p)
        tb.add_scalar("Small p/Failure Rate", failure_rate[i], p)
        tb.add_scalar("Small p/Asymptotic Fail", asymptotic_fail[i], p)
        tb.add_scalar("Small p/Asymptotic Success", asymptotic_success[i], p)
        tb.add_scalar("Small p/P_l", P_l[i], p)
        tb.add_scalar("Small p/Avg No Steps", average_number_of_steps_list[i], p)
        tb.add_scalar("Small p/P errors", p)



    tb.close()