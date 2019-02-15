import numpy as np

from Experiment_Engine.util import check_dict_else_default, check_attribute_else_default


class Puddle:

    def __init__(self, headX, headY, tailX, tailY, radius, length):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length

    def getDistance(self, xCoor, yCoor):
        u = (xCoor - self.tailX) * (self.headX - self.tailX) + (yCoor - self.tailY) * (self.headY - self.tailY) / self.length;

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor), 2) + np.power((self.tailY - yCoor), 2))
            else:
                dist = np.sqrt(np.power((self.headX - xCoor), 2) + np.power((self.headY - yCoor), 2))
        else:
            x = self.tailX + u * (self.headX - self.tailX)
            y = self.tailY + u * (self.headY - self.tailY)

            dist = np.sqrt(np.power((x - xCoor), 2) + np.power((y - yCoor), 2))

        if dist < self.radius:
            return self.radius - dist
        else:
            return 0


class PuddleWorld:
    """
    Environment Specifications:
    Number of Actions = 4
    Observation Dimension = 2 (x-coordinate, y-coordinate)
    Observation Dtype = np.float64
    Reward = -1 at every step + -400 * (distance to the nearest edge of each puddle)
        - This means that the farther into the puddle, the larger the penalty.

    Summary Name: steps_per_episode
    """

    def __init__(self, config, summary=None):
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             1000            The max number of actions executed before forcing
                                                                    a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        store_summary               bool            False           Whether to store the summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', 1000)
        self.norm_state = check_attribute_else_default(config, 'norm_state', True)
        self.store_summary = check_attribute_else_default(config, 'store_summary', default_value=False)
        self.summary = summary
        if self.store_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        self.num_action = 4
        self.num_state = 2

        """ Inner state of the environment """
        self.step_count = 0
        self.state = np.float64(np.random.uniform(low=0.0, high=0.1, size=(2,)))
        self.puddle1 = Puddle(0.45, 0.75, 0.10, 0.75, 0.1, 0.35)
        self.puddle2 = Puddle(0.45, 0.80, 0.45, 0.40, 0.1, 0.4)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0

        self.goalDimension = 0.05
        self.defDisplacement = 0.05

        self.goalXCoor = self.pworld_max_x - self.goalDimension
        self.goalYCoor = self.pworld_max_y - self.goalDimension

    def reset(self):
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.step_count)
        self.step_count = 0
        self.state = np.float64(np.random.uniform(low=0.0, high=0.1, size=(2,)))
        return self.get_current_state()

    def get_current_state(self):
        if self.norm_state:
            s = self.state
            s0 = (s[0] - 0.5) * 2.0
            s1 = (s[1] - 0.5) * 2.0
            return np.array([s0, s1], dtype=np.float64)
        else:
            s = self.state
            return np.array([s[0], s[1]], dtype=np.float64)

    def _terminal(self):
        s = self.state
        return bool((s[0] >= self.goalXCoor) and (s[1] >= self.goalYCoor))

    def _reward(self, x, y, terminal):
        if terminal:
            return -1
        reward = -1
        reward += (-400 * self.puddle1.getDistance(x, y))
        reward += (-400 * self.puddle2.getDistance(x, y))
        reward = reward
        return reward

    def step(self, a):
        self.step_count += 1
        s = self.state

        xpos = s[0]
        ypos = s[1]

        if a == 0:      # up
            ypos += self.defDisplacement
        elif a == 1:    # down
            ypos -= self.defDisplacement
        elif a == 2:    # right
            xpos += self.defDisplacement
        else:           # left
            xpos -= self.defDisplacement

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = np.float64(s)

        timeout = bool(self.step_count >= self.max_actions)
        terminal = self._terminal()
        reward = self._reward(xpos, ypos, terminal)

        return self.get_current_state(), reward, terminal, timeout
