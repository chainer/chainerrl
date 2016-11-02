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
from chainer import testing

from chainerrl.misc import random_seed
from chainerrl.experiments import train_agent


class _TestTraining(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, 'agent_final')
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
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)
            agent.replay_buffer.load(self.rbuf_filename)

        # Train
        train_agent.train_agent(
            agent=agent, env=env, steps=steps, outdir=self.tmpdir)

        # Test
        total_r = 0.0
        obs = env.reset()
        done = False
        reward = 0.0
        while not done:
            # s = np.expand_dims(obs, 0)
            # if gpu >= 0:
            #     s = cuda.to_gpu(s, device=gpu)
            action = agent.act(obs)
            if isinstance(action, cuda.cupy.ndarray):
                action = cuda.to_cpu(action)
            obs, reward, done, _ = env.step(action)
            total_r += reward
        agent.stop_episode()
        self.assertAlmostEqual(total_r, successful_return)

        # Save
        agent.save(self.agent_dirname)
        agent.replay_buffer.save(self.rbuf_filename)

    @testing.attr.slow
    def test_training_gpu(self):
        self._test_training(0, steps=3000)
        self._test_training(0, steps=300, load_model=True)

    @testing.attr.slow
    def test_training_cpu(self):
        self._test_training(-1, steps=3000)
