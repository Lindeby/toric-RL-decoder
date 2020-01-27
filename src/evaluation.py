def select_action_prediction(self, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
    # set network in eval mode
    self.policy_net.eval()
    # generate perspectives
    perspectives = self.toric.generate_perspective(grid_shift, self.toric.current_state)
    number_of_perspectives = len(perspectives)
    # preprocess batch of perspectives and actions 
    perspectives = Perspective(*zip(*perspectives))
    batch_perspectives = np.array(perspectives.perspective)
    batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
    batch_perspectives = batch_perspectives.to(self.device)
    batch_position_actions = perspectives.position
    # generate action value for different perspectives 
    with torch.no_grad():
        policy_net_output = self.policy_net(batch_perspectives)
        q_values_table = np.array(policy_net_output.cpu())
    #choose action using epsilon greedy approach
    rand = random.random()
    if(1 - epsilon > rand):
        # select greedy action 
        row, col = np.where(q_values_table == np.max(q_values_table))
        perspective = row[0]
        max_q_action = col[0] + 1
        step = Action(batch_position_actions[perspective], max_q_action)
        if prev_action == step:
            res = heapq.nlargest(2, q_values_table.flatten())
            row, col = np.where(q_values_table == res[1])
            perspective = row[0]
            max_q_action = col[0] + 1
            step = Action(batch_position_actions[perspective], max_q_action)
        q_value = q_values_table[row[0], col[0]]
    # select random action
    else:
        random_perspective = random.randint(0, number_of_perspectives-1)
        random_action = random.randint(1, number_of_actions)
        q_value = q_values_table[random_perspective, random_action-1]
        step = Action(batch_position_actions[random_perspective], random_action)

    return step, q_value


def prediction(self, num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, 
    show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, 
    print_Q_values=False, save_prediction=True):

    # load network for prediction and set eval mode 
    if PATH != None:
        self.load_network(PATH)
    self.policy_net.eval()

    # init matrices 
    ground_state_list = np.zeros(len(prediction_list_p_error))
    error_corrected_list = np.zeros(len(prediction_list_p_error))
    average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
    mean_q_list = np.zeros(len(prediction_list_p_error))
    failed_syndroms = []
    failure_rate = 0
    
    # loop through different p_error
    for i, p_error in enumerate(prediction_list_p_error):
        ground_state = np.ones(num_of_predictions, dtype=bool)
        error_corrected = np.zeros(num_of_predictions)
        mean_steps_per_p_error = 0
        mean_q_per_p_error = 0
        steps_counter = 0
        for j in range(num_of_predictions):
            num_of_steps_per_episode = 0
            prev_action = 0
            terminal_state = 0
            # generate random syndrom
            self.toric = Toric_code(self.system_size)

            if minimum_nbr_of_qubit_errors == 0:
                self.toric.generate_random_error(p_error)
            else:
                self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
            terminal_state = self.toric.terminal_state(self.toric.current_state)
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')
            
            init_qubit_state = deepcopy(self.toric.qubit_matrix)
            # solve syndrome
            while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1
                # choose greedy action
                action, q_value = self.select_action_prediction(number_of_actions=self.number_of_actions, 
                                                                epsilon=epsilon,
                                                                grid_shift=self.grid_shift,
                                                                prev_action=prev_action)
                prev_action = action
                self.toric.step(action)
                self.toric.current_state = self.toric.next_state
                terminal_state = self.toric.terminal_state(self.toric.current_state)
                mean_q_per_p_error = incremental_mean(q_value, mean_q_per_p_error, steps_counter)
                
                if plot_one_episode == True and j == 0 and i == 0:
                    self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))

            # compute mean steps 
            mean_steps_per_p_error = incremental_mean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
            # save error corrected 
            error_corrected[j] = self.toric.terminal_state(self.toric.current_state) # 0: error corrected # 1: error not corrected    
            # update groundstate
            self.toric.eval_ground_state()                                                          
            ground_state[j] = self.toric.ground_state # False non trivial loops

            if terminal_state == 1 or self.toric.ground_state == False:
                failed_syndroms.append(init_qubit_state)
                failed_syndroms.append(self.toric.qubit_matrix)

        success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)
        mean_q_list[i] = np.round(mean_q_per_p_error, 3)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, mean_q_list, failed_syndroms, ground_state_list, prediction_list_p_error, failure_rate
