import torch
from torch import from_numpy
import numpy as np
from src.util import generatePerspective, Action, Perspective


def dataToBatch(data, device):
    """ Converts data from the replay memory to appropriate dimensions.

    Params
    ======
    data: () Data from the replay buffer queue. Each item is a tuple of
                (('state', 'action', 'reward', 'next_state', 'terminal'), weight, index)
    Returns
    =======
    """

    def toNetInput(batch):
        batch_input = np.stack(batch, axis=0) 
        tensor = from_numpy(batch_input)
        tensor = tensor.type('torch.Tensor')
        return tensor.to(device)

    # get transitions and unpack them to minibatch
    batch = data[0]
    weights = data[1]
    index = data[2]
    weights = from_numpy(np.array(weights)).type('torch.Tensor').to(device)

    # preprocess batch_input and batch_target_input for the network
    list_state, list_action, list_reward, list_next_state, list_terminal = zip(*batch)

    batch_state = toNetInput(list_state)
    batch_next_state = toNetInput(list_next_state)

    # unpack action batch
    batch_action = Action(*zip(*list_action))
    batch_actions = np.array(batch_action.action) - 1 # -1 to compensate for env action index
    batch_actions = torch.Tensor(batch_actions).long()
    batch_actions = batch_actions.to(device)

    # preprocess batch_terminal and batch reward
    batch_terminal = from_numpy(np.array(list_terminal)).to(device)
    batch_reward   = from_numpy(np.array(list_reward)).type('torch.Tensor').to(device)

    return batch_state, batch_actions, batch_reward, batch_next_state, batch_terminal, weights, index

def predictMaxOptimized(model, batch_state, grid_shift, system_size, device):
    """ Generates the max Q values for a batch of states.
    Params
    ======
    model:          (torch.nn)
    batch_state     (List)
    grid_shift      (int)
    system_size:    (int)
    device:         (String){'cpu', 'cude'}

    Return
    ======
    (torch.Tensor) A tensor containing the max q value for each state.
    """
    
    master_batch_perspectives = []
    indices = []
    count_persp = 0
    largest_persp_batch = 0
    terminal_state_idx = []
    for i, state in enumerate(batch_state):
        # concat all perspectives to one batch, keep track of indices between batches
        perspectives = generatePerspective(int(system_size/2), system_size, np.array(state.cpu()))

        # no perspectives because terminal state
        if len(perspectives) == 0:
            terminal_state_idx.append(i)
            perspectives = np.zeros((1 , 2,system_size, system_size))
        else:
            perspectives = Perspective(*zip(*perspectives))
            perspectives = perspectives.perspective

        master_batch_perspectives.extend(perspectives)

        ind = len(perspectives)
        indices.append(count_persp + ind)
        largest_persp_batch = max(largest_persp_batch, ind)
        count_persp += ind

    master_batch_perspectives = from_numpy(np.array(master_batch_perspectives)).type('torch.Tensor').to(device)

    output = None
    q_values = None
    model.eval()
    with torch.no_grad():
        output = model(master_batch_perspectives)
        q_values = np.array(output.cpu())
    
    # split q-values back into batches
    q_values = np.split(q_values, indices[:-1])
    # pad with 0 so all batches get same size
    q_values = np.array([np.concatenate((batch, np.zeros((largest_persp_batch - len(batch), 3))), axis=0) for batch in q_values])
    q_max_idx = q_values.reshape(q_values.shape[0], -1).argmax(1)
    maxpos_vect = np.column_stack(np.unravel_index(q_max_idx, q_values[0,:,:].shape))

    persp_idx, action_idx = np.split(maxpos_vect, 2, axis=1)
    batch_idx = np.arange(len(maxpos_vect))
    batch_output = q_values[batch_idx, persp_idx.flatten(), action_idx.flatten()]

    batch_output[terminal_state_idx] = 0
    batch_output = from_numpy(np.array(batch_output)).type('torch.Tensor')

    return batch_output



def predictMax(model, batch_state, batch_size, grid_shift, system_size, device):
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
            perspectives = generatePerspective(grid_shift, system_size, np.array(batch_state[i].cpu())) 
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

