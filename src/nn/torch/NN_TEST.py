import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# neural network CNN with one fully connected layer
class NN_TEST_11(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        def conv_to_fully_connected(input_size, filter_size, padding, stride):
            return (input_size - filter_size + 2 * padding)/ stride + 1

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)

        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 120, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(120, 111, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(111, 104, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(104, 103, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(103, 90, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(90, 80 , kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(80, 73 , kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(73, 71 , kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(71, 64, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(obs_space.shape[1], 3, 0, 1) # hardcoded system size
        self.linear1 = nn.Linear(64*int(output_from_conv)**2, 3)

    @override(TorchModelV2)
    def forward(self, input_dict, s, seq_lens):
        state = input_dict["obs"]
        def pad_circular(x, pad):
            x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
            x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
            x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
            x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
            return x
        state = pad_circular(state, 1)
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.conv4(state))
        state = F.relu(self.conv5(state))
        state = F.relu(self.conv6(state))
        state = F.relu(self.conv7(state))
        state = F.relu(self.conv8(state))
        state = F.relu(self.conv9(state))
        state = F.relu(self.conv10(state))
        state = F.relu(self.conv11(state))
        n_features = np.prod(state.size()[1:])
        state = state.view(-1, n_features)
        output = self.linear1(state)
        return (output, s)

    # TODO: Look into this
    def value_function(self):
        return self._value_out
