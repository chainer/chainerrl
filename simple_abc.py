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
        return np.asarray([self._state], dtype=np.float32)

    @property
    def reward(self):
        if self._state == 3:
            # success
            return 1
        elif self._state == 4:
            return 0
        else:
            return 0

    def receive_action(self, action):
        if action == 0 and self._state == 0:
            # A
            self._state = 1
        elif action == 1 and self._state == 1:
            # B
            self._state = 2
        elif action == 2 and self._state == 2:
            # C
            self._state = 3
        else:
            self._state = 4

    def initialize(self):
        self._state = 0

    @property
    def is_terminal(self):
        return self._state == 3 or self._state == 4
