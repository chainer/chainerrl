from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA
import unittest

import numpy as np

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents.dqn import compute_value_loss
from chainerrl.agents.dqn import compute_weighted_value_loss
from chainerrl.agents.dqn import DQN


class TestDQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnDiscreteABCBoltzmann(base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = chainerrl.explorers.Boltzmann()
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnContinuousABC(base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100,
                   episodic_update=True)


def _huber_loss_1(a):
    if abs(a) < 1:
        return 0.5 * a ** 2
    else:
        return abs(a) - 0.5


class TestComputeWeightedValueLoss(unittest.TestCase):

    def setUp(self):
        pass

    def test(self):
        y = np.asarray([1.0, 2.0, 3.0], dtype='f')
        t = np.asarray([2.1, 2.2, 2.3], dtype='f')
        gt_huber_losses = np.asarray([_huber_loss_1(a) for a in y - t])

        mean_loss = compute_value_loss(
            y, t, clip_delta=True, batch_accumulator='mean').data
        self.assertAlmostEqual(mean_loss, gt_huber_losses.mean(), places=5)

        sum_loss = compute_value_loss(
            y, t, clip_delta=True, batch_accumulator='sum').data
        self.assertAlmostEqual(sum_loss, gt_huber_losses.sum(), places=5)

        w1 = np.asarray([1, 1, 1])

        mean_loss_w1 = compute_weighted_value_loss(
            y, t, clip_delta=True, batch_accumulator='mean',
            weights=w1).data
        self.assertAlmostEqual(mean_loss_w1, mean_loss, places=5)

        sum_loss_w1 = compute_weighted_value_loss(
            y, t, clip_delta=True, batch_accumulator='sum',
            weights=w1).data
        self.assertAlmostEqual(sum_loss_w1, sum_loss, places=5)

        wu = np.random.uniform(low=0, high=2, size=3).astype('f')
        mean_loss_wu = compute_weighted_value_loss(
            y, t, clip_delta=True, batch_accumulator='mean',
            weights=wu).data
        self.assertAlmostEqual(
            mean_loss_wu, (gt_huber_losses * wu).mean(), places=5)

        sum_loss_wu = compute_weighted_value_loss(
            y, t, clip_delta=True, batch_accumulator='sum',
            weights=wu).data
        self.assertAlmostEqual(
            sum_loss_wu, (gt_huber_losses * wu).sum(), places=5)
