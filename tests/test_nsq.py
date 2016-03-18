import os
import unittest

import numpy as np
import chainer
from chainer import optimizers

import q_function
import nstep_q_learning
import async
import simple_abc
import random_seed
import replay_buffer
from simple_abc import ABC


class TestNSQ(unittest.TestCase):

    def setUp(self):
        pass

    def test_abc(self):
        self._test_abc(1)
        self._test_abc(5)

    def _test_abc(self, t_max):

        nproc = 8

        def agent_func():
            n_actions = 3
            q_func = q_function.FCSIQFunction(1, n_actions, 10, 2)
            opt = optimizers.RMSprop(1e-3, eps=1e-2)
            opt.setup(q_func)
            return nstep_q_learning.NStepQLearning(q_func, opt, t_max,
                                                   0.9, 0.1,
                                                   i_target=10)

        def env_func():
            return simple_abc.ABC()

        def run_func(agent, env):
            total_r = 0
            episode_r = 0

            for i in xrange(5000):

                total_r += env.reward
                episode_r += env.reward

                action = agent.act(env.state, env.reward, env.is_terminal)

                if env.is_terminal:
                    print 'i:{} episode_r:{}'.format(i, episode_r)
                    episode_r = 0
                    env.initialize()
                else:
                    env.receive_action(action)

            print 'pid:{}, total_r:{}'.format(os.getpid(), total_r)

            return agent

        # Train
        final_agent = async.run_async(nproc, agent_func, env_func, run_func)

        # Test
        env = env_func()
        total_r = env.reward
        while not env.is_terminal:
            action = final_agent.q_function.sample_greedily_with_value(
                env.state.reshape((1,) + env.state.shape))[0][0]
            print 'state:', env.state, 'action:', action
            env.receive_action(action)
            total_r += env.reward
        self.assertAlmostEqual(total_r, 1)
