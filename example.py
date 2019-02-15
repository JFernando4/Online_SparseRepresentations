from Experiment_Engine.util import *
from Experiment_Engine.function_approximators import VanillaNeuralNetwork
from Experiment_Engine.agent import Agent
from Experiment_Engine.Environments import MountainCar
import numpy as np

exp_config = Config()
exp_config.store_summary = True
summary = {}

""" Parameters for the Environment """
exp_config.max_actions = 5000
exp_config.norm_state = True

""" Parameters for the Function Approximator """
exp_config.state_dims = 2
exp_config.num_actions = 3
exp_config.gamma = 1.0
exp_config.epsilon = 0.1
exp_config.optim = 'adam'
exp_config.lr = 0.001
exp_config.reg_factor = 0.1
exp_config.reg_method = 'none'

env = MountainCar(config=exp_config, summary=summary)
fa = VanillaNeuralNetwork(config=exp_config, gates='silu-silu', summary=summary)
rl_agent = Agent(environment=env, function_approximator=fa, config=exp_config, summary=summary)

for i in range(500):
    print("Episode number:", i+1)
    rl_agent.train(1)
    print('\tThe cumulative reward was:', summary['return_per_episode'][-1])
    print('\tThe cumulative loss was:', np.round(summary['cumulative_loss_per_episode'][-1], 2))
