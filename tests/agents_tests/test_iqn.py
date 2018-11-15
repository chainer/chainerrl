from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import unittest

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents import iqn


@testing.parameterize(*testing.product({
    'quantile_thresholds_N': [1, 5],
    'quantile_thresholds_N_prime': [1, 7],
}))
class TestIQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=iqn.CosineBasisLinearReLU(64, hidden_size),
            f=L.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return iqn.IQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            quantile_thresholds_N=self.quantile_thresholds_N,
            quantile_thresholds_N_prime=self.quantile_thresholds_N_prime,
        )


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'N': [1, 5],
    'N_prime': [1, 7],
    'huber_loss_threshold': [0.5, 1],
}))
class TestComputeEltwiseHuberQuantileLoss(unittest.TestCase):

    def test(self):
        batch_size = self.batch_size
        N = self.N
        N_prime = self.N_prime
        huber_loss_threshold = self.huber_loss_threshold

        # Overestimation is penalized proportionally to tau
        # Underestimation is penalized proportionally to (1-tau)
        y = np.random.normal(size=(batch_size, N)).astype('f')
        y_var = chainer.Variable(y)
        t = np.random.normal(size=(batch_size, N_prime)).astype('f')
        tau = np.random.uniform(size=(batch_size, N)).astype('f')

        loss = iqn.compute_eltwise_huber_quantile_loss(
            y_var, t, tau, huber_loss_threshold=huber_loss_threshold)
        y_var_b, t_b = F.broadcast(
            F.reshape(y_var, (batch_size, N, 1)),
            F.reshape(t, (batch_size, 1, N_prime)),
        )
        self.assertEqual(loss.shape, (batch_size, N, N_prime))
        huber_loss = F.huber_loss(
            y_var_b, t_b, delta=huber_loss_threshold, reduce='no')
        self.assertEqual(huber_loss.shape, (batch_size, N, N_prime))

        for i in range(batch_size):
            for j in range(N):
                for k in range(N_prime):
                    # loss is always positive
                    scalar_loss = loss[i, j, k]
                    scalar_grad = chainer.grad(
                        [scalar_loss], [y_var])[0][i, j]
                    self.assertGreater(scalar_loss.array, 0)
                    if y[i, j] > t[i, k]:
                        # y over-estimates t
                        # loss equals huber loss scaled by tau
                        correct_scalar_loss = tau[i, j] * huber_loss[i, j, k]
                    else:
                        # y under-estimates t
                        # loss equals huber loss scaled by (1-tau)
                        correct_scalar_loss = (
                            (1 - tau[i, j]) * huber_loss[i, j, k])
                    correct_scalar_grad = chainer.grad(
                        [correct_scalar_loss], [y_var])[0][i, j]
                    self.assertAlmostEqual(
                        scalar_loss.array,
                        correct_scalar_loss.array,
                        places=5,
                    )
                    self.assertAlmostEqual(
                        scalar_grad.array,
                        correct_scalar_grad.array,
                        places=5,
                    )
