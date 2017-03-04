from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import numpy as np

from chainerrl import env
from chainerrl import spaces


class ABC(env.Env):
    """Toy problem for only a testing purpose.

    The optimal policy is:
        state0 -> action0
        state1 -> action1
        state2 -> action2
    Observations are one-hot vectors that represents which state it is now.
    """

    def __init__(self, size=2, discrete=True, partially_observable=False,
                 episodic=True, deterministic=False):
        self.size = size
        self.terminal_state = size
        self.episodic = episodic
        self.partially_observable = partially_observable
        self.deterministic = deterministic
        self.n_max_offset = 1
        # (s_0, ..., s_N) + terminal state + offset
        self.n_dim_obs = self.size + 1 + self.n_max_offset
        self.observation_space = spaces.Box(
            low=np.asarray([-np.inf] * self.n_dim_obs, dtype=np.float32),
            high=np.asarray([np.inf] * self.n_dim_obs, dtype=np.float32))
        if discrete:
            self.action_space = spaces.Discrete(self.size)
        else:
            self.action_space = spaces.Box(
                low=-np.ones(self.size, dtype=np.float32),
                high=np.ones(self.size, dtype=np.float32))

    def observe(self):
        state_vec = np.zeros((self.n_dim_obs,), dtype=np.float32)
        state_vec[self._state + self._offset] = 1.0
        return state_vec

    def reset(self):
        self._state = 0
        if self.partially_observable:
            # For partially observable settings, observations are shifted by
            # episode-dependent some offsets.
            if self.deterministic:
                self._offset = ((getattr(self, '_offset', 0) + 1) %
                                (self.n_max_offset + 1))
            else:
                self._offset = np.random.randint(self.n_max_offset + 1)
        else:
            self._offset = 0
        return self.observe()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.clip(action,
                             self.action_space.low,
                             self.action_space.high)
            if self.deterministic:
                action = np.argmax(action)
            else:
                prob = np.exp(action) / np.exp(action).sum()
                action = np.random.choice(range(self.size), p=prob)
        reward = 0
        done = False
        if action == self._state:
            # Correct
            if self._state == self.size - 1:
                # Goal
                reward = 1.0
                if self.episodic:
                    # Terminal
                    done = True
                    self._state = self.terminal_state
                else:
                    # Restart
                    self._state = 0
            else:
                self._state += 1
        else:
            # Incorrect
            if self.episodic:
                # Terminal
                done = True
                self._state = self.terminal_state
        return self.observe(), reward, done, None

    def close(self):
        pass
