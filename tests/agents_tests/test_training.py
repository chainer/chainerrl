from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import logging
import os
import tempfile
import unittest

from chainer import testing

from chainerrl.experiments import train_agent_with_evaluation
from chainerrl.misc import random_seed


class _TestTraining(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, 'agent_final')
        self.rbuf_filename = os.path.join(self.tmpdir, 'rbuf.pkl')

    def make_agent(self, env, gpu):
        raise NotImplementedError()

    def make_env_and_successful_return(self, test):
        raise NotImplementedError()

    def _test_training(self, gpu, steps=5000, load_model=False):

        random_seed.set_random_seed(1)
        logging.basicConfig(level=logging.DEBUG)

        env, _ = self.make_env_and_successful_return(test=False)
        test_env, successful_return = self.make_env_and_successful_return(
            test=True)
        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)
            agent.replay_buffer.load(self.rbuf_filename)

        # Train
        train_agent_with_evaluation(
            agent=agent, env=env, steps=steps, outdir=self.tmpdir,
            eval_frequency=200, eval_n_runs=5, successful_score=1,
            eval_env=test_env)

        agent.stop_episode()

        # Test
        n_test_runs = 5
        for _ in range(n_test_runs):
            total_r = 0.0
            obs = test_env.reset()
            done = False
            reward = 0.0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = test_env.step(action)
                total_r += reward
            agent.stop_episode()
            self.assertAlmostEqual(total_r, successful_return)

        # Save
        agent.save(self.agent_dirname)
        agent.replay_buffer.save(self.rbuf_filename)

    @testing.attr.slow
    @testing.attr.gpu
    def test_training_gpu(self):
        self._test_training(0, steps=100000)
        self._test_training(0, steps=0, load_model=True)

    @testing.attr.slow
    def test_training_cpu(self):
        self._test_training(-1, steps=100000)
        self._test_training(-1, steps=0, load_model=True)
