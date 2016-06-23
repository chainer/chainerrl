from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy as np
import chainer
from chainer import optimizers
from chainer import functions as F

from links import dqn_head
import policy
from links import fc_tail_q_function
from links import fc_tail_v_function


def generate_different_two_states():
    """Generate differnet two input planes.

    Generated input two planes differ only at one random pixel.
    """
    sample_state = np.random.rand(1, 4, 84, 84).astype(np.float32)
    a = sample_state.copy()
    b = sample_state.copy()
    pos = np.random.randint(a.size)
    a.ravel()[pos] = 0.8
    b.ravel()[pos] = 0.2
    assert not np.allclose(a, b)
    return chainer.Variable(a), chainer.Variable(b)


class _TestDQNHead(unittest.TestCase):
    """Test that DQN heads are trainable."""

    def create_head(self):
        pass

    def create_optimizer(self):
        # return optimizers.RMSpropOnes(lr=1e-3, eps=1e-1)
        return optimizers.Adam()

    def test_v_function(self):
        head = self.create_head()
        v_func = fc_tail_v_function.FCTailVFunction(
            head, head.n_output_channels)
        opt = self.create_optimizer()
        opt.setup(v_func)
        a, b = generate_different_two_states()
        for _ in range(1000):
            # a
            v_func.zerograds()
            loss = (v_func(a) - 1.0) ** 2 / 2
            loss.backward()
            opt.update()
            # b
            v_func.zerograds()
            loss = (v_func(b) - 0.0) ** 2 / 2
            loss.backward()
            opt.update()

        va = float(v_func(a).data)
        vb = float(v_func(b).data)
        print((va, vb))
        self.assertAlmostEqual(va, 1.0, places=3)
        self.assertAlmostEqual(vb, 0.0, places=3)

    def test_q_function(self):
        n_actions = 3
        head = self.create_head()
        q_func = fc_tail_q_function.FCTailQFunction(
            head, head.n_output_channels, n_actions=n_actions)
        opt = self.create_optimizer()
        opt.setup(q_func)
        a, b = generate_different_two_states()
        action = np.random.randint(n_actions)
        for _ in range(1000):
            # a
            q_func.zerograds()
            loss = (q_func(a, [action]) - 1.0) ** 2 / 2
            loss.backward()
            opt.update()
            # b
            q_func.zerograds()
            loss = (q_func(b, [action]) - 0.0) ** 2 / 2
            loss.backward()
            opt.update()

        qa = float(q_func(a, [action]).data)
        qb = float(q_func(b, [action]).data)
        print((qa, qb))
        self.assertAlmostEqual(qa, 1.0, places=3)
        self.assertAlmostEqual(qb, 0.0, places=3)

    def test_policy(self):
        n_actions = 3
        head = self.create_head()
        pi = policy.FCSoftmaxPolicy(head.n_output_channels, n_actions)
        opt = self.create_optimizer()
        opt.setup(chainer.ChainList(head, pi))
        a, b = generate_different_two_states()

        def pout_func(s):
            return pi(head(s))

        def compute_loss(s, gt_label):
            pout = pout_func(s)
            return F.softmax_cross_entropy(
                pout.logits,
                chainer.Variable(np.asarray([gt_label], dtype=np.int32)))

        for _ in range(1000):
            loss_a = compute_loss(a, 0)
            loss_b = compute_loss(b, 1)
            print('loss for a:', loss_a.data)
            print('loss for b:', loss_b.data)
            head.zerograds()
            pi.zerograds()
            loss_a.backward()
            loss_b.backward()
            opt.update()

        pa = float(pi(head(a)).probs.data[0, 0])
        pb = float(pi(head(b)).probs.data[0, 1])
        self.assertAlmostEqual(pa, 1.0, places=3)
        self.assertAlmostEqual(pb, 1.0, places=3)


class TestNatureDQNHead(_TestDQNHead):

    def create_head(self):
        return dqn_head.NatureDQNHead()


class TestNIPSDQNHead(_TestDQNHead):

    def create_head(self):
        return dqn_head.NIPSDQNHead()
