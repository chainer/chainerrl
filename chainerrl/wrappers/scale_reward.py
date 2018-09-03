from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym


class ScaleReward(gym.RewardWrapper):
    """Scale reward by a scale factor.

    Args:
        env: Env to wrap.
        scale (float): Scale factor.

    Attributes:
        original_reward: Reward before casting.
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def _reward(self, reward):
        self.original_reward = reward
        return self.scale * reward
