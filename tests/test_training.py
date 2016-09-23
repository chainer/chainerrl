from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import tempfile
import unittest
import logging

from chainer import cuda

import random_seed
import run_dqn


class _TestTraining(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_filename = os.path.join(self.tmpdir, 'model.h5')
        self.rbuf_filename = os.path.join(self.tmpdir, 'rbuf.pkl')

    def make_agent(self, env, gpu):
        raise NotImplementedError()

    def make_env_and_successful_return(self):
        raise NotImplementedError()

    def _test_training(self, gpu, steps=5000, load_model=False):

        random_seed.set_random_seed(1)
        logging.basicConfig(level=logging.DEBUG)

        env, successful_return = self.make_env_and_successful_return()
        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load model from', self.model_filename)
            agent.load_model(self.model_filename)
            agent.replay_buffer.load(self.rbuf_filename)

        # Train
        run_dqn.run_dqn_with_evaluation(
            agent=agent, env=env, steps=steps, outdir=self.tmpdir,
            save_final_model=False)

        # Test
        total_r = 0.0
        obs = env.reset()
        done = False
        reward = 0.0
        agent.prepare_for_new_episode()
        while not done:
            # s = np.expand_dims(obs, 0)
            # if gpu >= 0:
            #     s = cuda.to_gpu(s, device=gpu)
            action = agent.select_greedy_action(obs)
            if isinstance(action, cuda.cupy.ndarray):
                action = cuda.to_cpu(action)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        self.assertAlmostEqual(total_r, successful_return)

        # Save
        agent.save_model(self.model_filename)
        agent.replay_buffer.save(self.rbuf_filename)

    def test_training_gpu(self):
        self._test_training(0, steps=3000)
        self._test_training(0, steps=300, load_model=True)

    def test_training_cpu(self):
        self._test_training(-1, steps=3000)
