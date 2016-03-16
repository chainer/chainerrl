import numpy as np
import environment


class DelayedXOR(environment.EpisodicEnvironment):
    """XOR with delayed rewards.

    There're two subgoals to achieve:
      1. Answer XOR correctly, which start a countdown
      2. Repeat choosing action 0 until the count decreased to one (choosing
         action 1 would increase the count, so it should not be chosen)
    The agent will receive a positive reward only if the count is one.
    Episodes are terminated if the count is one or n_delay * 2.
    """

    def __init__(self, n_delay):
        self.n_delay = n_delay
        self.initialize()

    @property
    def state(self):
        return self._state.astype(np.float32)

    @property
    def reward(self):
        return self.n_delay if self._state[2] == 1 else -0.01

    def receive_action(self, action):
        if self._state[2] == 0:
            # First you need to answer ax XOR problem
            if self._state[0] ^ self._state[1] == action:
                self._state[2] = self.n_delay
        else:
            # After answering the correct answer, you need to repeat action 0
            # to receive a reward
            if action == 0:
                self._state[2] -= 1
            else:
                self._state[2] += 1
        self._state[:2] = np.random.randint(2, size=2)

    def initialize(self):
        self._state = np.empty(3, dtype=np.int32)
        self._state[:2] = np.random.randint(2, size=2)
        self._state[2] = 0

    @property
    def is_terminal(self):
        return self._state[2] == 1 or self._state[2] == self.n_delay * 2
