from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym


class TREXReward(gym.RewardWrapper):
    """Implements Trajectory-ranked Reward EXtrapolation (TREX):

    https://arxiv.org/abs/1904.06387

    Args:
        env: Env to wrap.
        demos: A list of ranked demonstrations

    Attributes:
        demos: A list of demonstrations
    """

    def __init__(self, env, demos):
        super().__init__(env)
        self.demos = demos
        self.original_reward = None
        self._train()

    def reward(self, reward):
        self.original_reward = reward
        # trex_reward = self.trex_network(reward)
        trex_reward = reward
        return trex_reward

    def _train(self):
        pass
