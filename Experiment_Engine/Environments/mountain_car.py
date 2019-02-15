import numpy as np
from Experiment_Engine.util import check_dict_else_default, check_attribute_else_default, Config


class MountainCar:
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float64
    Reward = -1 at every step

    Summary Name: steps_per_episode
    """

    def __init__(self, config=None, summary=None):
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             1000            The max number of actions executed before forcing
                                                                    a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        store_summary               bool            False           Whether to store the summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', default_value=1000)
        self.norm_state = check_attribute_else_default(config, 'norm_state', default_value=True)
        self.store_summary = check_attribute_else_default(config, 'store_summary', default_value=False)
        self.summary = summary
        if self.store_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        " Inner state of the environment "
        self.step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.high = np.array([0.5, 0.07], dtype=np.float64)
        self.low = np.array([-1.2, -0.07], dtype=np.float64)
        self.action_dictionary = {0: -1,    # accelerate backwards
                                   1: 0,    # coast
                                   2: 1}    # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.step_count)
        self.step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state

    " Update environment "
    def step(self, A):
        self.step_count += 1

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[A]
        reward = -1.0
        terminate = False
        timeout = bool(self.step_count >= self.max_actions)

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * np.cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        self.current_state = np.array((position, velocity), dtype=np.float64)

        if self.norm_state:
            return self.normalize(self.current_state), reward, terminate, timeout
        else:
            return self.current_state, reward, terminate, timeout

    @staticmethod
    def normalize(state):
        """ normalize to [-1, 1] """
        temp_state = np.zeros(shape=2, dtype=np.float64)
        temp_state[0] = (state[0] + 0.35) / 0.85

        temp_state[1] = (state[1]) / 0.07
        return temp_state

    def get_current_state(self):
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state


if __name__ == "__main__":
    config = Config()
    config.store_summary = True
    config.max_actions = 100

    summary = {}
    actions = 3

    env = MountainCar(config, summary=summary)

    steps = 100
    cumulative_reward = 0
    terminations = 0
    successful_episode_steps = []

    for i in range(steps):
        action = np.random.randint(actions)
        old_state = env.get_current_state()
        new_state, reward, terminate, timeout = env.step(action)
        print("Old state:", np.round(old_state, 3), "-->",
              "Action:", action, "-->",
              "New state:", np.round(new_state, 3))
        cumulative_reward += reward
        if terminate or timeout:
            print("\n## Reset ##\n")
            if terminate:
                terminations += 1
                successful_episode_steps.append(env.step_count)
            env.reset()

    print("Number of steps per episode:", summary['steps_per_episode'])
    print("Number of episodes that reached the end:", terminations)
    average_length = np.average(successful_episode_steps) if len(successful_episode_steps) > 0 else np.inf
    print("The average number of steps per episode was:", average_length)
    print("Cumulative reward:", cumulative_reward)
