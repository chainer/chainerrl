from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import os
import tempfile
import unittest

from chainer import optimizers
from chainer import testing
import numpy as np

from chainerrl.agents.a3c import A3CSeparateModel
from chainerrl.agents.ppo import PPO
from chainerrl.envs.abc import ABC
from chainerrl.experiments import train_agent_with_evaluation
from chainerrl import policies
from chainerrl import v_functions


@testing.parameterize(*(
    testing.product({
        'clip_eps_vf': [None, 0.2],
        'lambd': [0.0, 0.5],
        'discrete': [False, True],
        'standardize_advantages': [False, True],
    })
))
class TestPPO(unittest.TestCase):

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

        env, _ = self.make_env_and_successful_return(test=False)
        test_env, successful_return = self.make_env_and_successful_return(
            test=True)
        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        train_agent_with_evaluation(
            agent=agent, env=env, steps=steps, outdir=self.tmpdir,
            eval_interval=200, eval_n_runs=50, successful_score=1,
            eval_env=test_env)

        agent.stop_episode()

        # Test
        n_test_runs = 100
        n_succeeded = 0
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
            if np.isclose(total_r, successful_return):
                n_succeeded += 1

        if require_success:
            self.assertGreater(n_succeeded, 0.8 * n_test_runs)

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        model = self.make_model(env)

        opt = optimizers.Adam(alpha=3e-4)
        opt.setup(model)

        return self.make_ppo_agent(env=env, model=model, opt=opt, gpu=gpu)

    def make_ppo_agent(self, env, model, opt, gpu):
        return PPO(model, opt, gpu=gpu, gamma=0.9, lambd=self.lambd,
                   update_interval=50, minibatch_size=25, epochs=3,
                   clip_eps_vf=self.clip_eps_vf,
                   standardize_advantages=self.standardize_advantages)

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

        return A3CSeparateModel(pi=pi, v=v)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=self.discrete, deterministic=test), 1
