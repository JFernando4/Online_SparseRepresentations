import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):

    def __init__(self, input_dims=1, hidden_layer_dims=1, output_layer_dims=1):
        super(Network, self).__init__()
        # input_dims = state dimensions
        # hidden_layer_dims = number of hidden neurons
        # output_layer_dims = number of actions
        self.fc1 = nn.Linear(input_dims, hidden_layer_dims, bias=True)
        self.fc2 = nn.Linear(hidden_layer_dims, output_layer_dims, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DqnAgent:

    def __init__(self, gamma=1, lr=0.00025, epsilon=0.1, network=Network()):
        self.epsilon = epsilon                                  # exploration parameter for epsilon-greedy policy
        self.gamma = torch.tensor(gamma, dtype=torch.float64)   # discount factor
        self.network = network                                  # neural network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def loss_function(self, minibatch):
        # computes the loss function:
        # cummulative_sum = 0
        # for sample in minibatch:
        #       delta = R_{t+1} + \max_a Q(S_{t+1}, a) - Q(S_t, A_t)    --- Temporal-Difference Error of Q-Learning
        #       cummulative_sum += delta ** 2
        # loss = cummulative_sum / size_of_minibatch                    --- Mean Squared TD Error
        current_state, current_action, next_reward, next_state = minibatch
        with torch.no_grad():
            current_action_value = self.network.forward(current_state)[current_action]

            next_action_value = self.network.forward(next_state)
            next_action_value = torch.argmax(next_action_value, dim=1)

        estimated_exp_return = next_reward + self.gamma * next_action_value
        td_error = estimated_exp_return - current_action_value
        ms_td_error = td_error.mean()
        return ms_td_error

    def training_step(self, minibatch):
        # applies gradient after sampling from the experience replay buffer
        self.optimizer.zero_grad()
        loss = self.loss_function(minibatch)
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        # chooses an action according to an epsilon-greedy policy
        with torch.no_grad():
            action_values = self.network.forward(state)

        p = np.random.rand()
        if p > self.epsilon:
            return torch.argmax(action_values, dim=1).numpy()
        else:
            return np.random.randint(0, action_values.size()[0] - 1, dtype=np.int8)


class ExperienceReplayBuffer:

    def __init__(self):
        pass

