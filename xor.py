import numpy as np
import environment


class XOR(environment.EpisodicEnvironment):

    def __init__(self):
        self.initialize()

    @property
    def state(self):
        return self._state.astype(np.float32)

    @property
    def reward(self):
        return self._reward

    def receive_action(self, action):
        if self._state[0] ^ self._state[1] == action:
            # Correct answer
            self._reward = 1
        else:
            # Incorrect answer
            self._reward = 0
        self._state = np.random.randint(2, size=2)

    def initialize(self):
        self._state = np.random.randint(2, size=2)
        self._reward = 0

    @property
    def is_terminal(self):
        return False
