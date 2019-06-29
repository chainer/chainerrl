from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym
import numpy as np


class RandomizeAction(gym.ActionWrapper):
    """Apply a random action instead of the one sent by the agent.

    This wrapper can be used to make a stochastic env. The common use is
    for evaluation in Atari environments, where actions are replaced with
    random ones with a low probability.

    Only gym.spaces.Discrete is supported as an action space.

    For exploration during training, use explorers like
    chainerrl.explorers.ConstantEpsilonGreedy instead of this wrapper.

    Args:
        env (gym.Env): Env to wrap.
        random_fraction (float): Fraction of actions that will be replaced
            with a random action. It must be in [0, 1].
    """

    def __init__(self, env, random_fraction):
        super().__init__(env)
        assert 0 <= random_fraction <= 1
        assert isinstance(env.action_space, gym.spaces.Discrete),\
            'RandomizeAction supports only gym.spaces.Discrete as an action space'  # NOQA
        self._random_fraction = random_fraction
        self._np_random = np.random.RandomState()

    def action(self, action):
        if self._np_random.rand() < self._random_fraction:
            return self._np_random.randint(self.env.action_space.n)
        else:
            return action

    def seed(self, seed):
        super().seed(seed)
        self._np_random.seed(seed)
