import numpy as np
import environment


class ABC(environment.EpisodicEnvironment):
    """Very simple toy problem.

    If the agent can choose actions 0, 1, 2 exactly in this order, it will receive reward 1. Otherwise, if it failed to do so, the environment is terminated with reward 0.
    """

    def __init__(self):
        self.initialize()

    @property
    def state(self):
        state_vec = np.zeros((5,), dtype=np.float32)
        state_vec[self._state] = 1.0
        return state_vec

    def initialize(self):
        self._state = 0

    @property
    def is_terminal(self):
        return self._state == 3 or self._state == 4

    def reset(self):
        self._state = 0
        return self.state

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
        return self.state, reward, self.is_terminal, None
