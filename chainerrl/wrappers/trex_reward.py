from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
import chainer.functions as F
import chainer.links as L
import gym
import numpy as np

from chainerrl.misc.batch_states import batch_states


class TREXNet(chainer.ChainList):
    """TREX's architecture: https://arxiv.org/abs/1904.06387"""


    def __init__(self):
        layers = [
            L.Convolution2D(4, 16, 7, stride=3),
            L.Convolution2D(16, 16, 5, stride=2),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Linear(784, 64),
            L.Linear(64, 1)
        ]

        super(TREXNet, self).__init__(*layers)

    def __call__(self, trajectory):
        h = trajectory
        for layer in self:
            h = F.leaky_relu(layer(h))
        return h


class TREXReward(gym.Wrapper):
    """Implements Trajectory-ranked Reward EXtrapolation (TREX):

    https://arxiv.org/abs/1904.06387.

    Args:
        env: Env to wrap.
        demos (RankedDemoDataset): A list of ranked demonstrations
        network: A reward network

    Attributes:
        demos: A list of demonstrations
        trex_network: Reward network

    """

    def __init__(self, env, ranked_demos, network=TREXNet()):
        super().__init__(env)
        self.ranked_demos = ranked_demos
        self.trex_network = network
        self.prev_reward = None
        self._train()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = batch_states([observation], self.trex_network.xp, self._phi)
        with chainer.no_backprop_mode():
            trex_reward = self.trex_network(obs)
        return observation, reward, done, info

    def _train(self):
        #TODOs: sample and align by time
        # subsample 6000 trajectory pairs between 50 and 100 observations long. 
        # We optimized the reward functions using Adam with a learning rate of 5e-5 for 30,000 steps.
        # enduro exception 
        pass

    def _phi(self, x):
        return np.asarray(x, dtype=np.float32) / 255
