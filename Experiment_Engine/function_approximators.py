import numpy as np
import torch

from Experiment_Engine.networks import TwoLayerFullyConnected, weight_init
from Experiment_Engine.util import *


class NeuralNetworkFunctionApproximation:

    def __init__(self, config, gates, summary=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions             int             3                   Number of actions available to the agent
        gamma                   float           1.0                 discount factor
        epsilon                 float           0.1                 exploration parameter
        state_dims              int             2                   number of dimensions of the environment's states
        optim                   str             'sgd'               optimization method. Choices: 'sgd', 'adam',
                                                                    'rmsprop'
        lr                      float           0.001               learning rate
        store_summary           bool            False               store the summary of the agent
                                                                    (cumulative_loss_per_episode)
        """
        assert isinstance(config, Config)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.epsilon = check_attribute_else_default(config, 'epsilon', 0.1)
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.optim = check_attribute_else_default(config, 'optim', 'sgd', choices=['sgd', 'adam', 'rmsprop'])
        self.lr = check_attribute_else_default(config, 'lr', 0.001)
        self.store_summary = check_attribute_else_default(config, 'store_summary', False)
        if self.store_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'cumulative_loss_per_episode', [])

        self.h1_dims = 32
        self.h2_dims = 256
        self.cumulative_loss = 0
        self.net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                          output_dims=self.num_actions, gates=gates)
        self.net.apply(weight_init)

        if self.optim == 'sgd': self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optim == 'adam': self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optim == 'rmsprop': self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

    def compute_return(self, reward, state, action, termination):
        # Computes the Sarsa(0) return. It assumes reward, action, and termination are all a single number.
        with torch.no_grad():
            av_function = self.net.forward(state)[action]
            next_step_bool = (1 - int(termination))
            sarsa_zero_return = reward + next_step_bool * self.gamma * av_function
        return sarsa_zero_return

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            with torch.no_grad():
                optim_action = self.net.forward(state).argmax().numpy()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def save_summary(self):
        if not self.store_summary:
            return
        self.summary['cumulative_loss_per_episode'].append(self.cumulative_loss)
        self.cumulative_loss = 0


class VanillaNeuralNetwork(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        gates                   str             relu-relu           types of gates for the network
        reg_factor              float           0.1                 factor for the regularization method
        reg_method              string          'none'              regularization method. Choices: 'none', 'l1', 'l2'
        """
        self.gates = check_attribute_else_default(config, 'gates', 'relu-relu')
        super(VanillaNeuralNetwork, self).__init__(config, gates=self.gates, summary=summary)
        self.reg_factor = check_attribute_else_default(config, 'reg_factor', 0.1)
        self.reg_method = check_attribute_else_default(config, 'reg_method', 'none',
                                                       choices=['none', 'l1', 'l2'])
        if self.reg_method == 'l1':
            self.reg_function = torch.abs
        elif self.reg_method == 'l2':
            self.reg_function = lambda z: torch.pow(z, 2)

    def update(self, state, action, reward, next_state, next_action, termination):
        # Performs an update to the parameters of the nn. It assumes action, reward, next_action, and termination are
        # a single number / boolean
        sarsa_zero_return = self.compute_return(reward, next_state, next_action, termination)
        self.optimizer.zero_grad()
        loss = (self.net(state)[action] - sarsa_zero_return) ** 2
        reg_loss = 0
        if self.reg_method != 'none':
            for name, param in self.net.named_parameters():
                reg_loss += torch.sum(self.reg_function(param))
        loss += self.reg_factor * reg_loss
        loss.backward()
        self.optimizer.step()
        if self.store_summary:
            self.cumulative_loss += loss.detach().numpy()


class ReplayBufferNeuralNetwork(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        gates                   str             relu-relu           types of gates for the network
        batch_size              int             32                  minibatch size
        training_step_count     int             0                   number of training steps so far
        tnet_update_freq        int             10                  the update frequency of the target network
        """
        assert isinstance(config, Config)
        self.config = config
        self.gates = check_attribute_else_default(self.config, 'gates', 'relu-relu')
        super(ReplayBufferNeuralNetwork, self).__init__(config, self.gates, summary)
        self.batch_size = check_attribute_else_default(config, 'batch_size', 32)
        self.training_step_count = check_attribute_else_default(config, 'training_step_count', 0)
        self.tnet_update_freq = check_attribute_else_default(config, 'tnet_update_freq', 10)
        self.replay_buffer = ReplayBuffer(config)
        self.target_net = TwoLayerFullyConnected(self.state_dims, h1_dims=self.h1_dims, h2_dims=self.h2_dims,
                                                 output_dims=self.num_actions, gates=self.gates)
        self.target_net.apply(weight_init)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            return

        self.training_step_count += 1
        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        sarsa_zero_return = self.compute_return(reward, next_state, next_action, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state).gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (prediction - sarsa_zero_return).pow(2).mean()
        loss.backward()
        self.optimizer.step()

        if self.store_summary:
            self.cumulative_loss += loss.detach().numpy()
        if (self.training_step_count % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def compute_return(self, reward, state, action, termination):
        with torch.no_grad():
            # av_function = torch.squeeze(self.target_net.forward(state).gather(1, torch.from_numpy(action).view(-1,1)))
            av_function = torch.max(self.target_net.forward(state), dim=1)[0]
            next_step_bool = torch.from_numpy((1 - np.int64(termination))).float()
            sarsa_zero_return = torch.from_numpy(reward).float() + next_step_bool * self.gamma * av_function
        return sarsa_zero_return


class ReplayBuffer:

    def __init__(self, config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        state_dims              int             2                   number of dimensions of the environment's state
        buffer_size             int             100                 size of the buffer
        """
        self.state_dims = check_attribute_else_default(config, 'state_dims', 2)
        self.buffer_size = check_attribute_else_default(config, 'buffer_size', 100)

        """ inner state """
        self.start = 0
        self.length = 0

        self.state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.action = np.empty(self.buffer_size, dtype=int)
        self.reward = np.empty(self.buffer_size, dtype=np.float64)
        self.next_state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.next_action = np.empty(self.buffer_size, dtype=int)
        self.termination = np.empty(self.buffer_size, dtype=bool)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise KeyError()
        elif isinstance(idx, np.ndarray):
            if (idx < 0).any() or (idx >= self.length).any():
                raise KeyError()
        shifted_idx = self.start + idx
        s = self.state.take(shifted_idx, axis=0, mode='wrap')
        a = self.action.take(shifted_idx, axis=0, mode='wrap')
        r = self.reward.take(shifted_idx, axis=0, mode='wrap')
        next_s = self.next_state.take(shifted_idx, axis=0, mode='wrap')
        next_a = self.next_action.take(shifted_idx, axis=0, mode='wrap')
        terminate = self.termination.take(shifted_idx, axis=0, mode='wrap')
        return s, a, r, next_s, next_a, terminate

    def store_transition(self, transition):
        if self.length < self.buffer_size:
            self.length += 1
        elif self.length == self.buffer_size:
            self.start = (self.start + 1) % self.buffer_size
        else:
            raise RuntimeError()

        storing_idx = (self.start + self.length - 1) % self.buffer_size
        state, action, reward, next_state, next_action, termination = transition
        self.state[storing_idx] = state
        self.action[storing_idx] = action
        self.reward[storing_idx] = reward
        self.next_state[storing_idx] = next_state
        self.next_action[storing_idx] = next_action
        self.termination[storing_idx] = termination

    def sample(self, sample_size):
        if sample_size > self.length or sample_size > self.buffer_size:
            raise ValueError("The sample size is to large.")
        sampled_idx = np.random.randint(0, self.length, sample_size)
        return self.__getitem__(sampled_idx)
