from src.util import load_network, incrementalMean, generatePerspective, Perspective, Action, convert_from_np_to_tensor
import numpy as np
from copy import deepcopy
import random, torch
import heapq


def evaluate(model, env, grid_shift, device, prediction_list_p_error, num_of_episodes=1,
    num_actions=3, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, 
    show_network=False, show_plot=False, minimum_nbr_of_qubit_errors=0, 
    print_Q_values=False, save_prediction=True):
    """ Evaluates the current policy by running some episodes.

    Params
    ======
    env:                            (gym.Env)
    model:                          (torch.nn)
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
    show_plot:                      (Bool)      (optional)
    minimum_nbr_of_qubit_errors:    (int)       (optional)
    print_Q_values:                 (Bool)      (optional)
    save_prediction:                (Bool)      (optional)

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
#    return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, failure_rate, prediction_list_p_error

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

            state = env.reset()
            
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                env.plot_toric_code(state, 'initial_syndrom')
            
            init_qubit_state = deepcopy(env.qubit_matrix)
            # solve syndrome
            while not terminal_state and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1
                
                # choose greedy action
                action, q_value = select_action_prediction( model=model,
                                                            device=device,
                                                            state=state,
                                                            toric_size=env.system_size,
                                                            number_of_actions=num_actions, 
                                                            epsilon=0,
                                                            grid_shift=grid_shift,
                                                            prev_action=prev_action)

                prev_action = action
                next_state, reward, terminal_state, _ = env.step(action)

                state = next_state
                mean_q_per_p_error = incrementalMean(q_value, mean_q_per_p_error, steps_counter)
                
                if plot_one_episode == True and j == 0 and i == 0:
                    env.plot_toric_code(state, 'step_'+str(num_of_steps_per_episode))

            # compute mean steps 
            mean_steps_per_p_error = incrementalMean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
            # save error corrected 
            error_corrected[j] = terminal_state # 1: error corrected # 0: error not corrected    
            
            # update groundstate
            ground_state[j] = env.evalGroundState()

            if terminal_state == 1 or ground_state[j] == False:
                failed_syndroms.append(init_qubit_state)
                failed_syndroms.append(env.qubit_matrix)

        success_rate = (num_of_episodes - np.sum(error_corrected)) / num_of_episodes # TODO: This does not make sense
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_episodes - np.sum(ground_state)) / num_of_episodes
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i] = np.round(mean_q_per_p_error, 3)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms



def select_action_prediction(model, device, state, toric_size, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
    # set network in eval mode
    model.eval()
    # generate perspectives
    perspectives = generatePerspective(grid_shift, toric_size, state)
    number_of_perspectives = len(perspectives)
    # preprocess batch of perspectives and actions 
    perspectives = Perspective(*zip(*perspectives))
    batch_perspectives = np.array(perspectives.perspective)
    batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
    batch_perspectives = batch_perspectives.to(device)
    batch_position_actions = perspectives.position
    # generate action value for different perspectives 
    with torch.no_grad():
        policy_net_output = model(batch_perspectives)
        q_values_table = np.array(policy_net_output.cpu())
    #choose action using epsilon greedy approach
    rand = random.random()
    if(1 - epsilon > rand):
        # select greedy action 
        row, col = np.where(q_values_table == np.max(q_values_table))
        perspective = row[0]
        max_q_action = col[0] + 1
        step = [  batch_position_actions[perspective][0],
                    batch_position_actions[perspective][1],
                    batch_position_actions[perspective][2],
                    max_q_action]

        if prev_action == step:
            res = heapq.nlargest(2, q_values_table.flatten())
            row, col = np.where(q_values_table == res[1])
            perspective = row[0]
            max_q_action = col[0] + 1
            step = [    batch_position_actions[perspective][0],
                        batch_position_actions[perspective][1],
                        batch_position_actions[perspective][2],
                        max_q_action]
        q_value = q_values_table[row[0], col[0]]
    # select random action
    else:
        random_perspective = random.randint(0, number_of_perspectives-1)
        random_action = random.randint(1, number_of_actions)
        q_value = q_values_table[random_perspective, random_action-1]
        action = [  batch_position_actions[random_perspective][0],
                    batch_position_actions[random_perspective][1],
                    batch_position_actions[random_perspective][2],
                    random_action]
    return step, q_value
