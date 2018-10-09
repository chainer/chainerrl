from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA
import unittest

import chainer
import chainer.links as L
from chainer import testing
import numpy as np

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents import categorical_dqn
from chainerrl.agents import CategoricalDQN


def _apply_categorical_projection_naive(y, y_probs, z):
    """Naively implemented categorical projection for checking results.

    See (7) in https://arxiv.org/abs/1802.08163.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    v_min = z[0]
    v_max = z[-1]
    xp = chainer.cuda.get_array_module(z)
    proj_probs = xp.zeros((batch_size, n_atoms), dtype=np.float32)
    for b in range(batch_size):
        for i in range(n_atoms):
            yi = y[b, i]
            p = y_probs[b, i]
            if yi <= v_min:
                proj_probs[b, 0] += p
            elif yi > v_max:
                proj_probs[b, -1] += p
            else:
                for j in range(n_atoms - 1):
                    if z[j] < yi <= z[j + 1]:
                        delta_z = z[j + 1] - z[j]
                        proj_probs[b, j] += (z[j + 1] - yi) / delta_z * p
                        proj_probs[b, j + 1] += (yi - z[j]) / delta_z * p
                        break
                else:
                    assert False
    return proj_probs


@testing.parameterize(
    *testing.product({
        'batch_size': [1, 7],
        'n_atoms': [2, 5],
        'v_range': [(-3, -1), (-2, 0), (-2, 1), (0, 1), (1, 5)],
    })
)
class TestApplyCategoricalProjectionToRandomCases(unittest.TestCase):

    def _test(self, xp):
        v_min, v_max = self.v_range
        z = xp.linspace(v_min, v_max, num=self.n_atoms, dtype=np.float32)
        y = xp.random.normal(
            size=(self.batch_size, self.n_atoms)).astype(np.float32)
        y_probs = xp.asarray(np.random.dirichlet(
            alpha=np.ones(self.n_atoms),
            size=self.batch_size).astype(np.float32))

        # Naive implementation as ground truths
        proj_gt = _apply_categorical_projection_naive(y, y_probs, z)
        # Projected probabilities should sum to one
        xp.testing.assert_allclose(
            proj_gt.sum(axis=1), xp.ones(self.batch_size, dtype=np.float32),
            atol=1e-5)

        # Batch implementation to test
        proj = categorical_dqn._apply_categorical_projection(y, y_probs, z)
        # Projected probabilities should sum to one
        xp.testing.assert_allclose(
            proj.sum(axis=1), xp.ones(self.batch_size, dtype=np.float32),
            atol=1e-5)

        # Both should be equal
        xp.testing.assert_allclose(proj, proj_gt, atol=1e-5)

    def test_cpu(self):
        self._test(np)

    @testing.attr.gpu
    def test_gpu(self):
        self._test(chainer.cuda.cupy)


class TestApplyCategoricalProjectionToManualCases(unittest.TestCase):

    def _test(self, xp):
        v_min, v_max = (-1, 1)
        n_atoms = 3
        z = xp.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        y = xp.asarray([
            [-1, 0, 1],
            [1, -1, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [0, 0, 0],
            [-0.5, 0, 1],
            [-0.5, 0, 0.5],
        ], dtype=np.float32)
        y_probs = xp.asarray([
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
        ], dtype=np.float32)

        proj_gt = xp.asarray([
            [0.5, 0.2, 0.3],
            [0.2, 0.3, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.45, 0.3],
            [0.25, 0.6, 0.15],
        ], dtype=np.float32)

        proj = categorical_dqn._apply_categorical_projection(y, y_probs, z)
        xp.testing.assert_allclose(proj, proj_gt, atol=1e-5)

    def test_cpu(self):
        self._test(np)

    @testing.attr.gpu
    def test_gpu(self):
        self._test(chainer.cuda.cupy)


def make_distrib_ff_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
        env.observation_space.low.size, env.action_space.n,
        n_atoms=n_atoms,
        v_min=v_min,
        v_max=v_max,
        n_hidden_channels=20,
        n_hidden_layers=1,
    )


def make_distrib_recurrent_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return chainerrl.links.Sequence(
        L.LSTM(env.observation_space.low.size, 20),
        chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
            20, env.action_space.n,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            n_hidden_channels=None,
            n_hidden_layers=0,
        ),
    )


class TestCategoricalDQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        return make_distrib_ff_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


# Continuous action spaces are not supported

class TestCategoricalDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_q_func(self, env):
        return make_distrib_recurrent_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            episodic_update=True)
