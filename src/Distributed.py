# standard libraries
from copy import deepcopy
from collections import namedtuple
from .ReplayMemory import PrioritizedReplayMemory
from .util import Transition, Action
# pytorch
from torch import from_numpy
import torch.distributed as dist
from torch.multiprocessing import Process, SimpleQueue


class Distributed():
    
    Perspective = namedtuple('Perspective', ['perspective', 'position'])
    Transition = namedtuple('Transition',['previous_state', 
                                          'action', 
                                          'reward', 
                                          'state', 
                                          'terminal']) 
    
    
    def __init__(self, policy_net, target_net, optimizer, env, replay_memory="proportional"):

        self.env = env
        self.optimizer = optimizer

        self.policy_net = policy_net
        self.target_net = target_net

        if replay_memory == "proportional":
            self.replay_memory = PrioritizedReplayMemory(100, 0.6) # TODO: temp size, alpha


    def train(train_steps, no_actors, learning_rate, epsilons, batch_size, policy_update):
        size = no_actors +1
        processes = []

        # Communication channels between processes
        weight_queue = SimpleQueue()
        transition_queue = SimpleQueue()
        
        args = {"no_actors": no_actors, "train_steps":train_steps, "batch_size":batch_size,
                "optimizer":self.optimizer,"policy_net":self.policy_net ,"target_net": self.target_net,
                "learning_rate":learning_rate,
                "policy_update":policy_update, "replay_memory":self.replay_memory}
        learner_process = Process(target=_init_process, args=(0, size, _learner, weight_queue,
                                                            transition_queue, args))
        learner_process.start()
        processes.append(learner_process)


        args = {"train_steps": train_steps, 
                "max_actions_per_episode":0, 
                "update_policy":policy_update,
                "size_local_memory_buffer":50, 
                "min_qubit_errors":0, 
                "model":deepcopy(self.policy_net),
                "env":self.env}
    
        for rank in range(no_actors):
            # a = new Actor()
            # a.run
            args["epsilon"] = epsilons[rank]
            actor_process = Process(target=_init_process, args=(rank+1, size, actor, weight_queue,
                                                                transition_queue, args))
            actor_process.start()
            processes.append(actor_process)

        for p in processes:
            p.join()

    def _init_process(rank, size, fn, wq, tq, args, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.2'
        os.environ['MASTER_PORT'] = '29501'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size, wq, tq, args)


    def _learner(self, rank, world_size, weight_queue, transition_queue, args):
        """The learner in a distributed RL setting. Updates the network params, pushes
        new network params to actors. Additionally, this function collects the transitions
        in the queue from the actors and manages the replay buffer.
        """

        def getBatches(batch_size):

            def toNetInput(batch, device):
                batch_input = np.stack(batch, axis=0) # not sure if it does anything
                # from np to tensor
                tensor = from_numpy(batch_input)
                tensor = tensor.type('torch.Tensor')
                return tensor.to(device)

            # get transitions and unpack them to minibatch
            # TODO: We want weights outside of the function
            transitions, weights, indices = replay_memory(batch_size, 0.4) # TODO: either self.replay memory or replay memory provided
            mini_batch = Transition(*zip(*transitions))

            # preprocess batch_input and batch_target_input for the network
            batch_state = toNetInput(mini_batch.state, self.device) # TODO: either self.device or device provided
            batch_next_state = toNetInput(mini_batch.next_state, self.device)

            # unpack action batch
            batch_actions = Action(*zip(*mini_batch.action))
            batch_actions = np.array(batch_actions.action) - 1
            batch_actions = torch.Tensor(batch_actions).long()
            batch_actions = batch_actions.to(self.device) 

            # preprocess batch_terminal and batch reward
            batch_terminal = convert_from_np_to_tensor(np.array(mini_batch.terminal)) 
            batch_terminal = batch_terminal.to(self.device)
            batch_reward = convert_from_np_to_tensor(np.array(mini_batch.reward))
            batch_reward = batch_reward.to(self.device)

            return batch_state, batch_actions, batch_reward, batch_next_state, weights, indices

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init counter
        push_new_weights = 0

        replay_memory = args["replay_memory"]
        policy_net = args["policy_net"]
        target_net = args["target_net"]

        # define criterion and optimizer
        criterion = nn.MSELoss(reduction='none')
        if args["optimizer"] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args["learning_rate"])
        elif args["optimizer"] == 'Adam':    
            optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])

        # Broadcast initial weights to actors
        group = dist.new_group([x for x in range(args["world_size"])])
        weights = parameters_to_vector(model.parameters())
        dist.broadcast(tensor=weights, src=rank, group=group) 
 
        # Wait until replay memory has enough transitions for one batch
        while len(replay_memory) < 16:
            if not transition_queue.empty():
                transition = transition_queue.get()
                replay_memory.save(transition) # Assuming memory entry generated by actor

        # Start training
        for t in range(train_steps):

            # Move to learner
            batch_state, batch_actions, batch_reward, batch_next_state , weights, indices = getBatches(batch_size)

            policy_net.train()
            target_net.eval()

            # compute policy net output
            policy_output = policy_net(batch_state)
            policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

            # compute target network output
            target_output = self.predict(self.target_net, batch_next_state, batch_size)
            target_output = target_output.to(device)
            
            # compute loss and update replay memory
            y = batch_reward + ((not batch_terminal) * self.discount_factor * target_output) # TODO: self.discount_factor or provide arg
            loss = self.getLoss(criterion, optimizer, y, policy_output, weights, indices) # Note: Also update priorites
            
            # backpropagate loss
            loss.backward()
            optimizer.step()

            # Get incomming transitions
            while not transition_queue.empty():
                transitions = replay_queue.get()
                replay_memory.save(transition) # Assuming memory entry generated by actor

            push_new_weights += 1
            if push_new_weights % args["policy_update"] == 0:
                weights = parameters_to_vector(policy.parameters())
                for actor in range(world_size-1):
                    weight_queue.put([weights.detach()])

                push_new_weights = 0

            # periodically evaluate network



    def getLoss(self, criterion, optimizer, y, output, weights, indices):
        loss = criterion(y, output)
        optimizer.zero_grad()
        # for prioritized experience replay
        if self.replay_memory == 'proportional': # TODO: self.replay_memory or provide arg
            tensor = from_numpy(np.array(weights))
            tensor = tensor.type('torch.Tensor')
            loss = tensor * loss.cpu() # TODO: Move to gpu
            priorities = torch.Tensor(loss, requires_grad=False)
            priorities = np.absolute(priorities.detach().numpy())
            self.memory.priority_update(indices, priorities)
        return loss.mean()

    def predict(self, net, batch_state, batch_size):
        """
        Params
        -------
        action_index: If the q value of the performed action is requested, 
        provide the chosen action index
        """
        net.eval()

        # Create containers
        batch_output = np.zeros(batch_size)
        batch_perspectives = np.zeros(shape=(batch_size, 2, self.system_size, self.system_size)) # TODO: either self.system_size or provide arg
        batch_actions = np.zeros(batch_size)

        for i in range(batch_size):
            if (batch_state[i].cpu().sum().item() == 0):
                batch_perspectives[i,:,:,:] = np.zeros(shape=(2, self.system_size, self.system_size))
            else:
                # Generate perspectives
                perspectives = generatePerspective(self.grid_shift, self.system_size, batch_state[i].cpu()) # TODO: either self.grid_shift or provide arg
                perspectives = Perspective(*zip(*perspectives))
                perspectives = np.array(perspectives.perspective)
                perspectives = from_numpy(perspectives)
                perspectives = tensor.type('torch.Tensor')
                perspectives = perspectives.to(self.device) # TODO either self.device or provide arg

                # prediction
                with torch.no_grad():
                    output = net(perspectives)
                    q_values = np.array(output.cpu())
                    row, col = np.where(q_values == np.max(q_values))
                    batch_output[i] = q_values[row[0], col[0]]      

                #perspective = perspectives[row[0]]
                #perspective = np.array(perspective.cpu())
                #batch_perspectives[i,:,:,:] = perspective
                #batch_actions[i] = col[0]

        batch_output = convert_from_np_to_tensor(batch_output)
        #batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        return batch_output#, batch_perspectives, batch_actions

                
    def _actor(self, rank, world_size, weight_queue, transition_queue, args):
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            # set network to eval mode
            model = args["model"]
            model.eval()

            env = args["env"]

            # Init network params
            weights = torch.zeros(1)
            dist.broadcast(tensor=weights, src=0)
            vector_to_parameters(weights, model.parameters())
            
            # init counters
            steps_counter = 0
            update_counter = 1
            # iteration = 0

            # local buffer of fixed size to store transitions before sending
            local_buffer = [None] * args["size_local_memory_buffer"]   
            local_memory_index = 0
            
            state = env.reset()
            steps_per_episode = 0
            terminal_state = False

            # main loop over training steps 
            for iteration in range(args["train_steps"]):
                #steps_counter += 1 # Just use iteration
                steps_per_episode += 1
                previous_state = state

                # select action using epsilon greedy policy
                action = self.select_action(number_of_actions=no_actions,
                                            epsilon=epsilon, 
                                            grid_shift=self.grid_shift,
                                            toric_size = env.system_size,
                                            state = state,
                                            model = model,
                                            device = device)

                state, reward, terminal_state, _ = env.step(action)

                # generate transition to stor in local memory buffer
                transition = generateTransition(action,
                                                    reward,
                                                    self.grid_shift, # self.gridshift???
                                                    previous_state,
                                                    state,
                                                    terminal_state)

                local_buffer.insert(local_memory_index, transition)
                if (local_memory_index % len(local_buffer)):
                    # TODO: Compute priorities
                    # TODO (Adam) send buffer to learner
                    local_memory_index = 0

                # if new weights are available, update network
                if not weight_queue.empty():
                    w = weight_queue.get()[0]
                    vector_to_parameters(w, model.parameters())

                if terminal_state or steps_per_episode > args["max_actions_per_episodes"]):
                        state = env.reset()
                        steps_per_episode = 0
                        terminal_state = False

                

    def select_action(self, number_of_actions, epsilon, grid_shift,
                      toric_size, state, model,device):
        # set network in evluation mode 
        model.eval()
        # generate perspectives 
        perspectives = self.generatePerspective(grid_shift, toric_size, state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(device)
        batch_position_actions = perspectives.position
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            with torch.no_grad():        
                policy_net_output = model(batch_perspectives)
                q_values_table = np.array(policy_net_output.cpu())
                row, col = np.where(q_values_table == np.max(q_values_table))
                perspective = row[0]
                max_q_action = col[0] + 1
                action = self.env.Action(batch_position_actions[perspective], max_q_action)
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            action = self.env.Action(batch_position_actions[random_perspective], random_action)  

        return action    
        

    def generatePerspective(self, grid_shift, toric_size, state):
        def mod(index, shift):
            index = (index + shift) % toric_size 
            return index
        perspectives = []
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        # qubit matrix 0
        for i in range(toric_size):
            for j in range(toric_size):
                if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
                plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    temp = Perspective(new_state, (0,i,j))
                    perspectives.append(temp)
        # qubit matrix 1
        for i in range(toric_size):
            for j in range(toric_size):
                if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
                plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    new_state = self.rotate_state(new_state) # rotate perspective clock wise
                    temp = Perspective(new_state, (1,i,j))
                    perspectives.append(temp)
        
        return perspectives

    def generateTransition(self, 
                            action, 
                            reward, 
                            grid_shift,       
                            previous_state,   #   Previous state before action
                            state,            #   Current state    
                            terminal_state    #   True/False
                            ):
        

        qubit_matrix = action.position[0]
        row = action.position[1]
        col = action.position[2]
        add_operator = action.action
        if qubit_matrix == 0:
            previous_perspective, perspective = shift_state(row, col, previous_state, state)
            action = Action((0, grid_shift, grid_shift), add_operator)
        elif qubit_matrix == 1:
            previous_perspective, perspective = shift_state(row, col)
            previous_perspective = self.rotate_state(previous_perspective)
            perspective = self.rotate_state(perspective)
            action = Action((1, grid_shift, grid_shift), add_operator)
        return Transition(previous_perspective, action, reward, perspective, terminal_state)

    def rotate_state(self, state):
         vertex_matrix = state[0,:,:]
         plaquette_matrix = state[1,:,:]
         rot_plaquette_matrix = np.rot90(plaquette_matrix)
         rot_vertex_matrix = np.rot90(vertex_matrix)
         rot_vertex_matrix = np.roll(rot_vertex_matrix, 1, axis=0)
         rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
         return rot_state 

        
    def shift_state(row, col, previous_state, state):
         previous_perspective = np.roll(previous_state, grid_shift-row, axis=1)
         previous_perspective = np.roll(previous_perspective, grid_shift-col, axis=2)
         perspective = np.roll(state, grid_shift-row, axis=1)
         perspective = np.roll(perspective, grid_shift-col, axis=2)
         return previous_perspective, perspective
