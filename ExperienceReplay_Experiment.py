import numpy as np
import argparse
import os
import torch
import pickle

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Acrobot, MountainCar, PuddleWorld             # environments
from Experiment_Engine import Agent, ReplayBufferNeuralNetwork              # agent and function approximator

NUMBER_OF_EPISODES = 500


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        self.run_results_dir = run_results_dir
        self.tnet_update_Freq = check_attribute_else_default(experiment_parameters, 'tnet_update_freq', 1)
        self.buffer_size = check_attribute_else_default(experiment_parameters, 'buffer_size', 10000)
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'acrobot', 'puddle_world'])
        self.verbose = experiment_parameters.verbose

        environment_dictionary = {
            'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3},
            'acrobot': {'class': Acrobot, 'state_dims': 4, 'num_actions': 3},
            'puddle_world': {'class': PuddleWorld, 'state_dims': 2, 'num_actions': 4}
                                  }

        self.config = Config()
        self.config.store_summary = True
        self.summary = {}

        """ Parameters for the Environment """
        self.config.max_actions = 5000
        self.config.norm_state = True

        """ Parameters for the Function Approximator """
        self.config.state_dims = environment_dictionary[self.environment_name]['state_dims']
        self.config.num_actions = environment_dictionary[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.lr = 0.001
        self.config.batch_size = 32
        self.config.training_step_count = 0
        self.config.buffer_size = self.buffer_size
        self.config.tnet_update_freq = self.tnet_update_Freq
        self.config.gates = 'relu-relu'

        self.env = environment_dictionary[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = ReplayBufferNeuralNetwork(config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        saving_times = [50, 100, 250, 500]
        for i in range(NUMBER_OF_EPISODES):
            episode_number = i + 1
            self.rl_agent.train(1)
            if self.verbose:
                print("Episode Number:", episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
                print('\tThe cumulative loss was:', np.round(self.summary['cumulative_loss_per_episode'][-1], 2))
            if episode_number in saving_times:
                self.save_network_params(suffix=str(episode_number)+'episodes')
        self.save_run_summary()

    def save_network_params(self, suffix='50episodes.pt'):
        params_path = os.path.join(self.run_results_dir, 'network_weights_'+suffix)
        torch.save(self.fa.net.state_dict(), params_path)

    def save_run_summary(self):
        summary_path = os.path.join(self.run_results_dir, 'summary.p')
        with open(summary_path, mode='wb') as summary_file:
            pickle.dump(self.summary, summary_file)
        config_path = os.path.join(self.run_results_dir, 'config.p')
        with open(config_path, mode='wb') as config_file:
            pickle.dump(self.config, config_file)


if __name__ == '__main__':
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', action='store', default='mountain_car', type=str, choices=['mountain_car', 'acrobot',
                                                                                           'puddle_world', 'catcher'])
    parser.add_argument('-tnet_update_freq', action='store', default=1, type=np.int64)
    parser.add_argument('-buffer_size', action='store', default=10000, type=np.int64)
    parser.add_argument('-verbose', action='store_true')
    exp_parameters = parser.parse_args()

    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.isdir(results_parent_directory):
        os.makedirs(results_parent_directory)

    environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env, 'experience_replay')
    if not os.path.isdir(environment_result_directory):
        os.makedirs(environment_result_directory)

    parameters_name = 'BufferSize' + str(exp_parameters.buffer_size) + "_Freq" + str(exp_parameters.tnet_update_freq)
    parameters_result_directory = os.path.join(environment_result_directory, parameters_name)
    if not os.path.isdir(parameters_result_directory):
        os.makedirs(parameters_result_directory)

    last_agent_number = len(os.listdir(parameters_result_directory))
    agent_id = 'agent_' + str(last_agent_number + 1)
    run_results_directory = os.path.join(parameters_result_directory, agent_id)
    os.makedirs(run_results_directory)

    experiment = Experiment(experiment_parameters=exp_parameters, run_results_dir=run_results_directory)
    experiment.run()


