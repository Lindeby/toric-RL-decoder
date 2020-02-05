# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import from_numpy
# other
import numpy as np

# from file
from src.util import Action, Perspective, Transition, generatePerspective
import time

#def learner(rank, world_size, weight_queue, transition_queue, args):
def learner(rank, world_size, args):
    """The learner in a distributed RL setting. Updates the network params, pushes
    new network params to actors. Additionally, this function collects the transitions
    in the queue from the actors and manages the replay buffer.
    """

    """ 
    args = {"no_actors",
            "train_steps",
            "batch_size",
            "optimizer",
            "policy_net",
            "target_net",
            "learning_rate",
            "device",
            "policy_update",
            "replay_memory",
            "discount_factor",
            "con_send_weights",
            "transition_queue_from_memory",
            update_priorities_queue_to_memory
            }
    """
    update_priorities_queue_to_memory = args["update_priorities_queue_to_memory"]
    transition_queue_from_memory = args["transition_queue_from_memory"]
    con_send_weights = args["con_send_weights"]
    transition_queue = args["transition_queue"]
    device = args["device"]
    replay_memory = args["replay_memory"]
    train_steps = args["train_steps"]
    policy_net = args["policy_net"]
    target_net = args["target_net"]
    discount_factor = args["discount_factor"]

    def dataToBatch(data):
        """ Converts data from the replay memory to appropriate dimensions.

        Params
        ======
        data: () Data from the replay buffer queue. Each item is a tuple of
                 (('state', 'action', 'reward', 'next_state', 'terminal'), index)
                 What about weights? Not needed since we compute new priorities.
                 What about indices? Needed for updating of priorities.

        Returns
        =======
        """

        def toNetInput(batch):
            batch_input = np.stack(batch, axis=0) 
            # from np to tensor
            tensor = from_numpy(batch_input)
            tensor = tensor.type('torch.Tensor')
            return tensor.to(device)

        # get transitions and unpack them to minibatch
        #transitions, weights, indices = replay_memory.sample(batch_size, 0.4)
        #batch = (*zip(*data[0]))
        #indices = (*zip(*data[1]))
        batch, index = zip(*data)
        # preprocess batch_input and batch_target_input for the network
        list_state, list_action, list_reward, list_next_state, list_terminal = zip(*batch)
        batch_state = toNetInput(list_state)
        #batch_state = toNetInput(batch[0])
        batch_next_state = toNetInput(list_next_state)
        #batch_next_state = toNetInput(batch[3])

        # unpack action batch
        batch_actions = np.array(list_action)
        batch_actions = torch.Tensor(batch_actions)
        batch_actions = batch_actions.to(device)
        #batch_actions = Action(*zip(*batch[1]))
        #batch_actions = np.array(batch_actions.action) - 1
        #batch_actions = torch.Tensor(batch_actions).long()
        #batch_actions = batch_actions.to(device) 

        # preprocess batch_terminal and batch reward
        batch_terminal = from_numpy(np.array(list_terminal)).type('torch.Tensor').to(device)
        #batch_terminal = from_numpy(np.array(batch[4])).type('torch.Tensor').to(device)
        batch_reward   = from_numpy(np.array(list_reward)).type('torch.Tensor').to(device)
        #batch_reward   = from_numpy(np.array(batch[2])).type('torch.Tensor').to(device)

        return batch_state, batch_actions, batch_reward, batch_next_state, indices

    # init counter
    push_new_weights = 0

    # define criterion and optimizer
    criterion = nn.MSELoss(reduction='none')
    if args["optimizer"] == 'RMSprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args["learning_rate"])
    elif args["optimizer"] == 'Adam':    
        optimizer = optim.Adam(policy_net.parameters(), lr=args["learning_rate"])
    

    # Push initial network params
    # weights = parameters_to_vector(policy_net.parameters()) 
    weights = policy_net.state_dict()
    for actor in range(world_size-2):
        con_send_weights[actor].send(weights)
    
    while True:
        time.sleep(1)

    # Wait until replay memory has enough transitions for one batch
    while transition_queue_from_memory.empty():
        continue

    # Start training
    for t in range(train_steps):
        print("learner: traning step: ",t," / ",train_steps)

        data = transition_queue_from_memory.get()

        # TODO: Everything out from here should be tenors, except indices
        batch_state, batch_actions, batch_reward, batch_next_state, indices = dataToBatch(data)

        policy_net.train()
        target_net.eval()

        # compute policy net output
        policy_output = policy_net(batch_state)
        policy_output = policy_output.gather(1, batch_actions.view(-1, 1)).squeeze(1)

        # compute target network output
        target_output = predictMax(target_net, batch_next_state, batch_size)
        target_output = target_output.to(device)
        
        # compute loss and update replay memory
        y = batch_reward + ((not batch_terminal) * discount_factor * target_output)
        loss = criterion(y, output)
        
        # Compute priotities # TODO Verify
        priorities = loss.cpu() # biased
        w = (1/len(priorities)*1/priorities)**args["beta"]
        w = w*(1/torch.max(w)) # normalise
        priorities *= w # weighted priorities

        optimizer.zero_grad()
        loss = loss.mean()

        # backpropagate loss
        loss.backward()
        optimizer.step()

        update_priorities_queue_to_memory.put([*zip(priorities, indices)])

        push_new_weights += 1
        if push_new_weights >= args["policy_update"]:
            weights = parameters_to_vector(policy_net.parameters())
            for actor in range(world_size-2):
                con_send_weights[actor].send(weights.detach())
            push_new_weights = 0

        # TODO: periodically evaluate network


# def getLoss(self, criterion, optimizer, y, output, weights, indices):
#     loss = criterion(y, output)
#     optimizer.zero_grad()
#     # # for prioritized experience replay
#     # if self.replay_memory == 'proportional':
#     #     tensor = from_numpy(np.array(weights))
#     #     tensor = tensor.type('torch.Tensor')
#     #     loss = tensor * loss.cpu() # TODO: Move to gpu
#     #     priorities = torch.Tensor(loss, requires_grad=False)
#     #     priorities = np.absolute(priorities.detach().numpy())
#     #     self.memory.priority_update(indices, priorities)
#     return loss.mean()


def predictMax(model, batch_state, grid_shift, system_size, device):
    """ Generates the max Q values for a batch of states.
    Params
    ======
    action_index: If the q value of the performed action is requested, 
    provide the chosen action index

    Return
    ======
    (torch.Tensor) A tensor containing the max q value for each state.
    """
    
    model.eval()

    # Create containers
    batch_output = np.zeros(batch_size)
    batch_perspectives = np.zeros(shape=(batch_size, 2, system_size, system_size))
    batch_actions = np.zeros(batch_size)

    for i in range(batch_size):
        if (batch_state[i].cpu().sum().item() == 0):
            batch_perspectives[i,:,:,:] = np.zeros(shape=(2, system_size, system_size))
        else:
            # Generate perspectives
            perspectives = generatePerspective(grid_shift, system_size, np.array(batch_state[i])) 
            perspectives = Perspective(*zip(*perspectives))
            perspectives = np.array(perspectives.perspective)
            perspectives = from_numpy(perspectives).type('torch.Tensor').to(device)

            # prediction
            with torch.no_grad():
                output = model(perspectives)
                q_values = np.array(output.cpu())
                row, col = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape) 
                batch_output[i] = q_values[row, col]      


    batch_output = from_numpy(batch_output).type('torch.Tensor')
    return batch_output
