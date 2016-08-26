from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import gym
gym.undo_logger_setup()
from gym import spaces

import env


class ABC(env.Env):
    """Very simple toy problem.

    If the agent can choose actions 0, 1, 2 exactly in this order, it will receive reward 1. Otherwise, if it failed to do so, the environment is terminated with reward 0.
    """

    def __init__(self, discrete=True):
        if discrete:
            self.action_space = spaces.Discrete(3)
        else:
            n_dim_action = 1
            self.action_space = spaces.Box(
                low=np.asarray([-0.49] * n_dim_action, dtype=np.float32),
                high=np.asarray([2.49] * n_dim_action, dtype=np.float32))

    def observe(self):
        state_vec = np.zeros((5,), dtype=np.float32)
        state_vec[self._state] = 1.0
        return state_vec

    def initialize(self):
        self._state = 0

    def is_terminal(self):
        return self._state == 3 or self._state == 4

    def reset(self):
        self._state = 0
        return self.observe()

    def step(self, action):
        if isinstance(action, np.ndarray):
            if action.size > 1:
                action = action[0]
            action = np.around(action)
        if action == 0 and self._state == 0:
            # A
            self._state = 1
            reward = 0.1
        elif action == 1 and self._state == 1:
            # B
            self._state = 2
            reward = 0.0
        elif action == 2 and self._state == 2:
            # C
            self._state = 3
            reward = 0.9
        else:
            self._state = 4
            reward = 0.0
        return self.observe(), reward, self.is_terminal(), None

    def close(self):
        pass
