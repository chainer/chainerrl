from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import tempfile
import unittest

from chainer import optimizers
from chainer import testing
import numpy as np

import chainerrl
from chainerrl.agents.a2c import A2C
from chainerrl.agents.a2c import A2CSeparateModel
from chainerrl.envs.abc import ABC
from chainerrl.experiments.evaluator import batch_run_evaluation_episodes
from chainerrl import policies
from chainerrl import v_functions


@testing.parameterize(*(
    testing.product({
        'num_processes': [1, 3],
        'use_gae': [False, True],
        'discrete': [False, True]
    })
))
class TestA2C(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, 'agent_final')

    @testing.attr.slow
    def test_abc_cpu(self):
        self._test_abc()
        self._test_abc(steps=0, load_model=True)

    @testing.attr.slow
    @testing.attr.gpu
    def test_abc_gpu(self):
        self._test_abc(gpu=0)

    def test_abc_fast_cpu(self):
        self._test_abc(steps=100, require_success=False)
        self._test_abc(steps=0, require_success=False, load_model=True)

    @testing.attr.gpu
    def test_abc_fast_gpu(self):
        self._test_abc(steps=100, require_success=False, gpu=0)

    def _test_abc(self, steps=1000000,
                  require_success=True, gpu=-1, load_model=False):

        env, _ = self.make_env_and_successful_return(
            test=False, n=self.num_processes)
        test_env, successful_return = self.make_env_and_successful_return(
            test=True, n=1)
        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        chainerrl.experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            log_interval=10,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=50,
            successful_score=1,
            eval_env=test_env,
        )
        env.close()

        # Test
        n_test_runs = 100
        eval_returns = batch_run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
        )
        test_env.close()
        n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
        if require_success:
            self.assertGreater(n_succeeded, 0.8 * n_test_runs)

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        model = self.make_model(env)

        opt = optimizers.Adam(alpha=3e-4)
        opt.setup(model)

        return self.make_a2c_agent(env=env, model=model, opt=opt, gpu=gpu,
                                   num_processes=self.num_processes)

    def make_a2c_agent(self, env, model, opt, gpu, num_processes):
        return A2C(model, opt, gpu=gpu, gamma=0.99,
                   num_processes=num_processes,
                   use_gae=self.use_gae)

    def make_model(self, env):
        n_hidden_channels = 50

        n_dim_obs = env.observation_space.low.size
        v = v_functions.FCVFunction(
            n_dim_obs,
            n_hidden_layers=2,
            n_hidden_channels=n_hidden_channels)

        if self.discrete:
            n_actions = env.action_space.n

            pi = policies.FCSoftmaxPolicy(
                n_dim_obs, n_actions,
                n_hidden_layers=2,
                n_hidden_channels=n_hidden_channels)
        else:
            n_dim_actions = env.action_space.low.size

            pi = policies.FCGaussianPolicy(
                n_dim_obs, n_dim_actions,
                n_hidden_layers=2,
                n_hidden_channels=n_hidden_channels)

        return A2CSeparateModel(pi=pi, v=v)

    def make_env_and_successful_return(self, test, n):

        def make_env():
            return ABC(discrete=self.discrete, deterministic=test)

        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [make_env for _ in range(n)])
        return vec_env, 1
