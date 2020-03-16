from src.util import load_network, incrementalMean, generatePerspectiveOptimized, Perspective, Action, convert_from_np_to_tensor
from src.util_actor import selectAction

import numpy as np
from copy import deepcopy
import random, torch
import heapq, gym, gym_ToricCode


def evaluate(model, env, env_config, grid_shift, device, prediction_list_p_error, num_of_episodes=1,
    num_actions=3, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, 
    show_network=False, minimum_nbr_of_qubit_errors=0, 
    print_Q_values=False):
    """ Evaluates the current policy by running some episodes.

    Params
    ======
    model:                          (torch.nn)
    env:                            (String)
    env_config:                     (dict)
    grid_shift:                     (int)
    device:                         (String) {"cpu", "cuda"}
    prediction_list_p_error:        (list)
    num_predictions:                (int)       (optional)
    num_actions:                    (int)       (optional)
    epsilon:                        (float)     (optional)
    num_of_steps:                   (int)       (optional)
    PATH:                           (String)    (optional)
    plot_one_episode:               (Bool)      (optional)
    show_network:                   (Bool)      (optional)
    minimum_nbr_of_qubit_errors:    (int)       (optional)
    print_Q_values:                 (Bool)      (optional)

    Return
    ======
    (list): success rate for each probability of error.
    (list): rate of which ground states was left
    (list): average number of steps
    (list): mean q-value
    (list): syndroms that failed to complete
    (list): list of floats representing the probability of generating an error
            in the environment for which the model was tested on.
    """

    # load network for prediction and set eval mode 
    if PATH != None:
        model = load_network(PATH)
    model.to(device)
    model.eval()

    # init matrices 
    ground_state_list = np.zeros(len(prediction_list_p_error))
    error_corrected_list = np.zeros(len(prediction_list_p_error))
    average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
    mean_q_list = np.zeros(len(prediction_list_p_error))
    failed_syndroms = []
    # failure_rate = 0

    cfg = {"size":env_config["size"], "min_qubit_errors":env_config["min_qubit_errors"], "p_error":prediction_list_p_error[0]}
    env = gym.make(env, config=cfg)
    
    # loop through different p_error
    for i, p_error in enumerate(prediction_list_p_error):
        ground_state = np.ones(num_of_episodes, dtype=bool)
        error_corrected = np.zeros(num_of_episodes)
        mean_steps_per_p_error = 0
        mean_q_per_p_error = 0
        steps_counter = 0

        for j in range(num_of_episodes):
            num_of_steps_per_episode = 0
            prev_action = 0
            terminal_state = 0

            state = env.reset(p_error=p_error)
            
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                env.plotToricCode(state, 'initial_syndrom')
            
            init_qubit_state = deepcopy(env.qubit_matrix)

            # solve syndrome
            energy_toric = []
            experimental_q_values = []
            while not terminal_state and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1
                
                # choose greedy action
                action, q_values = selectAction(number_of_actions=3,
                                                epsilon=1,
                                                grid_shift=grid_shift,
                                                toric_size=env.system_size,
                                                state=state,
                                                model=model,
                                                device=device)
                # action, q_value = select_action_prediction( model=model,
                #                                             device=device,
                #                                             state=state,
                #                                             toric_size=env.system_size,
                #                                             number_of_actions=num_actions, 
                #                                             epsilon=0,
                #                                             grid_shift=grid_shift,
                #                                             prev_action=prev_action)
                q_value = q_values[action-1]

                # prev_action = action
                next_state, reward, terminal_state, _ = env.step(action)
                
                experimental_q_values.append(q_value)
                energy_toric.append(np.sum(state) - np.sum(next_state)) 

                state = next_state
                mean_q_per_p_error = incrementalMean(q_value, mean_q_per_p_error, steps_counter)
                
                if plot_one_episode == True and j == 0 and i == 0:
                    env.plotToricCode(state, +str(num_of_steps_per_episode))

            theoretical_q_value = compute_theoretical_q_value(energy_toric)
            
            # compute mean steps 
            mean_steps_per_p_error = incrementalMean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
            # save error corrected 
            error_corrected[j] = terminal_state # 1: error corrected # 0: error not corrected    
            
            # update groundstate
            ground_state[j] = env.evalGroundState()

            if not terminal_state or ground_state[j] == False:
                failed_syndroms.append(init_qubit_state)
                failed_syndroms.append(env.qubit_matrix)

        success_rate = (num_of_episodes - np.sum(error_corrected)) / num_of_episodes
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_episodes - np.sum(ground_state)) / num_of_episodes
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i] = np.round(mean_q_per_p_error, 3)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, theoretical_q_value, experimental_q_values



# def select_action_prediction(model, device, state, toric_size, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
#     # set network in eval mode
#     model.eval()
#     # generate perspectives
#     perspectives, position = generatePerspectiveOptimized(grid_shift, toric_size, state)
#     number_of_perspectives = len(perspectives)
#     # preprocess batch of perspectives and actions 
#     batch_perspectives = convert_from_np_to_tensor(np.array(perspectives))
#     batch_perspectives = batch_perspectives.to(device)
#     batch_position_actions = np.array(position)
#     # generate action value for different perspectives 
#     with torch.no_grad():
#         policy_net_output = model(batch_perspectives)
#         q_values_table = np.array(policy_net_output.cpu())
    
#     #choose action using epsilon greedy approach
#     rand = random.random()
#     if(1 - epsilon > rand):
#         # select greedy action 
#         row, col = np.where(q_values_table == np.max(q_values_table))
#         p = row[0]
#         a = col[0] + 1
#         action = [  batch_position_actions[p][0],
#                     batch_position_actions[p][1],
#                     batch_position_actions[p][2],
#                     a]

#         # if prev_action == action:
#         #     res = heapq.nlargest(2, q_values_table.flatten())
#         #     row, col = np.where(q_values_table == res[1])
#         #     p = row[0]
#         #     a = col[0] + 1
#     # select random action
#     else:
#         p = random.randint(0, number_of_perspectives-1)
#         a = random.randint(0, number_of_actions-1) +1

#     q_value = q_values_table[p, a-1]
#     action = [  batch_position_actions[p][0],
#                 batch_position_actions[p][1],
#                 batch_position_actions[p][2],
#                 a]
    
#     return action, q_value


def compute_theoretical_q_value(energy_toric):
    energy_toric[-1] = 100
    energy_toric = energy_toric[::-1]

    q = np.zeros(len(energy_toric))
    gamma_array = [0.95**i for i in range(len(energy_toric))]

    for i in range(len(energy_toric)):
        i += 1
        actions_temp = energy_toric[0:i]
        gamma_temp = gamma_array[0:i]
        gamma_temp = np. array(gamma_temp[::-1])
        temp = np.sum(gamma_temp * actions_temp)
        q[i-1] = temp

    q = q[::-1]
    return q