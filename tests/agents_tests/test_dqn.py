from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA
import unittest

from chainer import testing
import numpy as np

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents.dqn import compute_value_loss
from chainerrl.agents.dqn import compute_weighted_value_loss
from chainerrl.agents.dqn import DQN

from basetest_training import _TestBatchTrainingMixin


class TestDQNOnDiscreteABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)

    def test_replay_capacity_checked(self):
        env, _ = self.make_env_and_successful_return(test=False)
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = chainerrl.replay_buffer.ReplayBuffer(capacity=90)
        with self.assertRaises(ValueError):
            self.make_dqn_agent(env=env, q_func=q_func, opt=opt,
                                explorer=explorer, rbuf=rbuf, gpu=None)


class TestDQNOnDiscreteABCBoltzmann(
        _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = chainerrl.explorers.Boltzmann()
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnContinuousABC(
        _TestBatchTrainingMixin, base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


# Batch training with recurrent models is currently not supported
class TestDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100,
                   episodic_update=True)


class TestNStepDQNOnDiscreteABC(base._TestNStepDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestNStepDQNOnDiscreteABCBoltzmann(base._TestNStepDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = chainerrl.explorers.Boltzmann()
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestNStepDQNOnContinuousABC(base._TestNStepDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


def _huber_loss_1(a):
    if abs(a) < 1:
        return 0.5 * a ** 2
    else:
        return abs(a) - 0.5


@testing.parameterize(
    *testing.product({
        'batch_accumulator': ['mean', 'sum'],
        'clip_delta': [True, False],
    })
)
class TestComputeValueLoss(unittest.TestCase):

    def setUp(self):
        self.y = np.asarray([1.0, 2.0, 3.0, 4.0], dtype='f')
        self.t = np.asarray([2.1, 2.2, 2.3, 2.4], dtype='f')
        if self.clip_delta:
            self.gt_losses = np.asarray(
                [_huber_loss_1(a) for a in self.y - self.t])
        else:
            self.gt_losses = np.asarray(
                [0.5 * a ** 2 for a in self.y - self.t])

    def test_not_weighted(self):
        loss = compute_value_loss(
            self.y, self.t, clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator).array
        if self.batch_accumulator == 'mean':
            gt_loss = self.gt_losses.mean()
        else:
            gt_loss = self.gt_losses.sum()
        self.assertAlmostEqual(loss, gt_loss, places=5)

    def test_uniformly_weighted(self):

        # Uniform weights
        w1 = np.ones(self.y.size, dtype='f')

        loss_w1 = compute_weighted_value_loss(
            self.y, self.t, clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator,
            weights=w1).array
        if self.batch_accumulator == 'mean':
            gt_loss = self.gt_losses.mean()
        else:
            gt_loss = self.gt_losses.sum()
        self.assertAlmostEqual(loss_w1, gt_loss, places=5)

    def test_randomly_weighted(self):

        # Random weights
        wu = np.random.uniform(low=0, high=2, size=self.y.size).astype('f')

        loss_wu = compute_weighted_value_loss(
            self.y, self.t, clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator,
            weights=wu).array
        if self.batch_accumulator == 'mean':
            gt_loss = (self.gt_losses * wu).mean()
        else:
            gt_loss = (self.gt_losses * wu).sum()
        self.assertAlmostEqual(loss_wu, gt_loss, places=5)
