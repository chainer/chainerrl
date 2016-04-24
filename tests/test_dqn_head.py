import unittest

import numpy as np
import chainer
from chainer import optimizers
from chainer import functions as F

import dqn_head
import fc_tail_q_function
import fc_tail_v_function
import fc_tail_policy


def generate_different_two_states():
    sample_state = np.random.rand(4, 84, 84).astype(np.float32)
    a = sample_state.copy()
    b = sample_state.copy()
    pos = np.random.randint(a.size)
    a.ravel()[pos] = 0.8
    b.ravel()[pos] = 0.2
    assert not np.allclose(a, b)
    return a, b


class _TestDQNHead(unittest.TestCase):
    """Test DQN heads are trainable
    """

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
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
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
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
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
        pi = fc_tail_policy.FCTailPolicy(
            head, head.n_output_channels, n_actions=n_actions)
        opt = self.create_optimizer()
        opt.setup(pi)
        a, b = generate_different_two_states()
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        for _ in range(1000):
            # a
            pi.zerograds()
            loss = F.softmax_cross_entropy(
                pi.forward(a),
                chainer.Variable(np.asarray([0], dtype=np.int32)))
            loss.backward()
            opt.update()
            # b
            pi.zerograds()
            loss = F.softmax_cross_entropy(
                pi.forward(b),
                chainer.Variable(np.asarray([1], dtype=np.int32)))
            loss.backward()
            opt.update()

        pa = float(pi(a, [0]).data)
        pb = float(pi(b, [1]).data)
        self.assertAlmostEqual(pa, 1.0, places=3)
        self.assertAlmostEqual(pb, 1.0, places=3)


class TestNatureDQNHead(_TestDQNHead):

    def create_head(self):
        return dqn_head.NatureDQNHead()


class TestNIPSDQNHead(_TestDQNHead):

    def create_head(self):
        return dqn_head.NIPSDQNHead()
