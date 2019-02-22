import numpy as np
import argparse
import os
import torch
import pickle

from Experiment_Engine.util import check_attribute_else_default, Config     # utilities
from Experiment_Engine import Acrobot, MountainCar, PuddleWorld             # environments
from Experiment_Engine import Agent, DistRegNeuralNetwork                   # agent and function approximator

NUMBER_OF_EPISODES = 500


class Experiment:

    def __init__(self, experiment_parameters, run_results_dir):
        """" Parameters of the experiment given by the user """
        self.run_results_dir = run_results_dir
        self.environment_name = check_attribute_else_default(experiment_parameters, 'env', 'mountain_car',
                                                             choices=['mountain_car', 'acrobot', 'puddle_world'])
        self.learning_rate = check_attribute_else_default(exp_parameters, 'lr', 0.001,
                                                          choices=[0.01, 0.004, 0.001, 0.00025, 0.0000625])
        self.reg_factor = check_attribute_else_default(experiment_parameters, 'reg_factor', 0.01,
                                                       choices=[0.0, 0.1, 0.01, 0.001])
        self.beta = check_attribute_else_default(experiment_parameters, 'beta', 0.1,
                                                 choices=[0.1, 0.2, 0.5])
        self.ma_alpha = check_attribute_else_default(experiment_parameters, 'ma_alpha', 0.1,
                                                     choices=[0.1, 0.01])
        self.verbose = experiment_parameters.verbose

        """ Experiment set up """
        environment_dictionary = {
            'mountain_car': {'class': MountainCar, 'state_dims': 2, 'num_actions': 3},
            'acrobot': {'class': Acrobot, 'state_dims': 4, 'num_actions': 3},
            'puddle_world': {'class': PuddleWorld, 'state_dims': 2, 'num_actions': 4}
                                  }
        self.config = Config()
        self.config.store_summary = True
        self.summary = {}

        """ Parameters for the Environment """
        self.config.max_actions = 2000
        self.config.norm_state = True

        """ Parameters for the Function Approximator """
        self.config.state_dims = environment_dictionary[self.environment_name]['state_dims']
        self.config.num_actions = environment_dictionary[self.environment_name]['num_actions']
        self.config.gamma = 1.0
        self.config.epsilon = 0.1
        self.config.optim = "adam"
        self.config.lr = self.learning_rate
        self.config.reg_factor = self.reg_factor
        self.config.beta = self.beta
        self.config.ma_alpha = self.ma_alpha

        self.env = environment_dictionary[self.environment_name]['class'](config=self.config, summary=self.summary)
        self.fa = DistRegNeuralNetwork(config=self.config, summary=self.summary)
        self.rl_agent = Agent(environment=self.env, function_approximator=self.fa, config=self.config,
                              summary=self.summary)

    def run(self):
        saving_times = [50, 100, 250, 500]
        for i in range(NUMBER_OF_EPISODES):
            episode_number = i + 1
            self.rl_agent.train(1)
            if self.verbose and (((i+1) % 10 == 0) or i == 0):
                print("Episode Number:", episode_number)
                print('\tThe cumulative reward was:', self.summary['return_per_episode'][-1])
                print('\tThe cumulative loss was:', np.round(self.summary['cumulative_loss_per_episode'][-1], 2))
            if episode_number in saving_times:
                self.save_network_params(suffix=str(episode_number)+'episodes')
        self.save_run_summary()

    def save_network_params(self, suffix='50episodes'):
        params_path = os.path.join(self.run_results_dir, 'network_weights_' + suffix + '.pt')
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
    parser.add_argument('-env', action='store', default='mountain_car', type=str,
                        choices=['mountain_car', 'acrobot', 'puddle_world'])
    parser.add_argument('-lr', action='store', default=0.001, type=np.float64,
                        choices=[0.01, 0.004, 0.001, 0.00025, 0.0000625])
    parser.add_argument('-reg_factor', action='store', default=0.1, type=float,
                        choices=[0.1, 0.01, 0.001])
    parser.add_argument('-beta', action='store', default=0.1, type=np.float64,
                        choices=[0.1, 0.2, 0.5])
    parser.add_argument('-ma_alpha', action='store', default=0.1, type=np.float64,
                        choices=[0.1, 0.01])
    parser.add_argument('-verbose', action='store_true')
    exp_parameters = parser.parse_args()

    """ This part creates the directory: working_dir/Results/environment/method/parameter_combination/agent_number"""
    # General results directory
    results_parent_directory = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(results_parent_directory):
        os.makedirs(results_parent_directory)
    # Directory specific to the environment and the method
    environment_result_directory = os.path.join(results_parent_directory, exp_parameters.env,
                                                'distributional_regularizers')
    if not os.path.exists(environment_result_directory):
        os.makedirs(environment_result_directory)
    # Directory specific to the parameters
    parameters_name = 'LearningRate' + str(exp_parameters.lr) \
                      + '_RegFactor' + str(exp_parameters.reg_factor) \
                      + '_Beta' + str(exp_parameters.beta) \
                      + '_MovingAvgAlpha' + str(exp_parameters.ma_alpha)
    parameters_result_directory = os.path.join(environment_result_directory, parameters_name)
    if not os.path.exists(parameters_result_directory):
        os.makedirs(parameters_result_directory)
    # Directory specific to the run
    last_agent_number = len(os.listdir(parameters_result_directory))
    agent_id = 'agent_' + str(last_agent_number + 1)
    run_results_directory = os.path.join(parameters_result_directory, agent_id)
    os.makedirs(run_results_directory)

    """ Setting up and running the experiment """
    experiment = Experiment(experiment_parameters=exp_parameters, run_results_dir=run_results_directory)
    experiment.run()

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
# ma_alpha = {0.1, 0.01}
