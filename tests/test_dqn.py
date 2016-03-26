import unittest

from chainer import optimizers

import q_function
from dqn import DQN
import random_seed
import replay_buffer
from simple_abc import ABC


class TestDQN(unittest.TestCase):

    def setUp(self):
        pass

    def _test_abc(self, gpu):

        random_seed.set_random_seed(0)

        q_func = q_function.FCSIQFunction(1, 3, 10, 2)

        opt = optimizers.RMSpropGraves(
            lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-4)
        opt.setup(q_func)

        rbuf = replay_buffer.ReplayBuffer(1e5)
        agent = DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, epsilon=0.1,
                    replay_start_size=1000, target_update_frequency=1000)

        env = ABC()

        total_r = 0
        episode_r = 0

        # Train
        for i in xrange(5000):
            episode_r += env.reward
            total_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                print 'i:{} epsilon:{} episode_r:{}'.format(i, agent.epsilon, episode_r)
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)

        # Test
        agent.epsilon = 0.0
        env.initialize()
        total_r = env.reward
        while not env.is_terminal:
            action = agent.act(env.state, env.reward, env.is_terminal)
            env.receive_action(action)
            total_r += env.reward
        self.assertAlmostEqual(total_r, 1)

    def test_abc_gpu(self):
        self._test_abc(0)

    def test_abc_cpu(self):
        self._test_abc(-1)
