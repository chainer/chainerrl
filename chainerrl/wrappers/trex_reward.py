from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import gym
import numpy as np

from chainerrl.misc.batch_states import batch_states


def random_subseq(seq, subseq_len, start=None):
    if len(seq) <= subseq_len:
        return seq
    else:
        i = np.random.randint(0, len(seq) - subseq_len + 1)
        return seq[i:i + subseq_len]


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
        env: a ScoreMaskEnv
        ranked_demos (RankedDemoDataset): A list of ranked demonstrations
        steps: number of gradient steps
        sub_traj_len: a tuple containing (min, max) traj length to sample
        network: A reward network to train

    Attributes:
        demos: A list of demonstrations
        trex_network: Reward network

    """

    def __init__(self, env,
                 ranked_demos,
                 steps=30000,
                 sub_traj_len=(50,100),
                 optimizer=optimizers.Adam(),
                 network=TREXNet()):
        super().__init__(env)
        self.ranked_demos = ranked_demos
        self.steps = steps
        self.trex_network = network
        self.opt = optimizer
        self.opt.setup(self.trex_network)
        self.prev_reward = None
        self.mask = env.mask
        self._train()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = batch_states([observation], self.trex_network.xp, self._phi)
        with chainer.no_backprop_mode():
            trex_reward = self.trex_network(obs)
        return observation, trex_reward, done, info

    def _train(self):
        # construct a dataset of 6000 pairs from the ranked demos
        # be sure to sample them appropriately (i.e. using start times), and random lengths
        # TODO: Check how original paper handles the length being too long for episode. (if at all
        for _ in range(self.steps):
            pass
        # use the mask
        #TODOs: sample and align by time
        # subsample 6000 trajectory pairs between 50 and 100 observations long. 
        # We optimized the reward functions using Adam with a learning rate of 5e-5 for 30,000 steps.
        # enduro exception 
        # at the end save the network

    def _phi(self, x):
        return np.asarray(x, dtype=np.float32) / 255
