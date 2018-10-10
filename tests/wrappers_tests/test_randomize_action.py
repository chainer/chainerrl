from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

from chainer import testing
from chainer.testing import condition
import gym
import gym.spaces

import chainerrl


class ActionRecordingEnv(gym.Env):

    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gym.spaces.Discrete(3)

    def __init__(self):
        self.past_actions = []

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        self.past_actions.append(action)
        return self.observation_space.sample(), 0, False, {}


@testing.parameterize(*testing.product({
    'random_fraction': [0, 0.3, 0.6, 1],
}))
class TestRandomizeAction(unittest.TestCase):

    @condition.retry(3)
    def test_action_ratio(self):
        random_fraction = self.random_fraction
        env = ActionRecordingEnv()
        env = chainerrl.wrappers.RandomizeAction(
            env, random_fraction=random_fraction)
        env.reset()
        n = 1000
        delta = 0.05
        for _ in range(n):
            # Always send action 0
            env.step(0)
        # Ratio of selected actions should be:
        #   0: (1 - random_fraction) + random_fraction/3
        #   1: random_fraction/3
        #   2: random_fraction/3
        self.assertAlmostEqual(
            env.env.past_actions.count(0) / n,
            (1 - random_fraction) + random_fraction / 3, delta=delta)
        self.assertAlmostEqual(
            env.env.past_actions.count(1) / n,
            random_fraction / 3, delta=delta)
        self.assertAlmostEqual(
            env.env.past_actions.count(2) / n,
            random_fraction / 3, delta=delta)

    @condition.retry(3)
    def test_seed(self):

        def get_actions(seed):
            random_fraction = self.random_fraction
            env = ActionRecordingEnv()
            env = chainerrl.wrappers.RandomizeAction(
                env, random_fraction=random_fraction)
            env.seed(seed)
            for _ in range(1000):
                # Always send action 0
                env.step(0)
            return env.env.past_actions

        a_seed0 = get_actions(0)
        a_seed1 = get_actions(1)
        b_seed0 = get_actions(0)
        b_seed1 = get_actions(1)

        self.assertEqual(a_seed0, b_seed0)
        self.assertEqual(a_seed1, b_seed1)
        if self.random_fraction > 0:
            self.assertNotEqual(a_seed0, a_seed1)
