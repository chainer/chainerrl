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

import chainer
import chainer.functions as F
from chainer import links as L
from chainer import optimizers
from chainer import testing
import numpy as np

import chainerrl
from chainerrl.envs.abc import ABC
from chainerrl.experiments.evaluator import batch_run_evaluation_episodes
from chainerrl.experiments.evaluator import run_evaluation_episodes
from chainerrl.experiments import train_agent_batch_with_evaluation
from chainerrl.experiments import train_agent_with_evaluation


def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)


@testing.parameterize(*(
    testing.product({
        'episodic': [False, True],
    })
))
class TestTD3(unittest.TestCase):

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

    @testing.attr.slow
    def test_abc_batch_cpu(self):
        self._test_abc_batch()
        self._test_abc_batch(steps=0, load_model=True)

    @testing.attr.slow
    @testing.attr.gpu
    def test_abc_batch_gpu(self):
        self._test_abc_batch(gpu=0)

    def test_abc_batch_fast_cpu(self):
        self._test_abc_batch(steps=100, require_success=False)
        self._test_abc_batch(steps=0, require_success=False, load_model=True)

    @testing.attr.gpu
    def test_abc_batch_fast_gpu(self):
        self._test_abc_batch(steps=100, require_success=False, gpu=0)

    def _test_abc(self, steps=100000,
                  require_success=True, gpu=-1, load_model=False):

        env, _ = self.make_env_and_successful_return(test=False)
        test_env, successful_return = self.make_env_and_successful_return(
            test=True)

        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        max_episode_len = None if self.episodic else 2

        # Train
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=test_env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=successful_return,
            train_max_episode_len=max_episode_len,
        )

        agent.stop_episode()

        # Test
        n_test_runs = 5
        eval_returns = run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
            max_episode_len=max_episode_len,
        )
        if require_success:
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            self.assertEqual(n_succeeded, n_test_runs)

        # Save
        agent.save(self.agent_dirname)

    def _test_abc_batch(self, steps=100000,
                        require_success=True, gpu=-1, load_model=False):

        env, _ = self.make_vec_env_and_successful_return(test=False)
        test_env, successful_return = self.make_vec_env_and_successful_return(
            test=True)

        agent = self.make_agent(env, gpu)

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        max_episode_len = None if self.episodic else 2

        # Train
        train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            eval_env=test_env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=successful_return,
            max_episode_len=max_episode_len,
        )
        env.close()

        # Test
        n_test_runs = 5
        eval_returns = batch_run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
            max_episode_len=max_episode_len,
        )
        test_env.close()
        if require_success:
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            self.assertEqual(n_succeeded, n_test_runs)

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        obs_size = env.observation_space.low.size
        action_size = env.action_space.low.size
        hidden_size = 20
        policy = chainer.Sequential(
            L.Linear(obs_size, hidden_size),
            F.relu,
            L.Linear(hidden_size, action_size,
                     initialW=chainer.initializers.LeCunNormal(1e-1)),
            F.tanh,
            chainerrl.distribution.ContinuousDeterministicDistribution,
        )
        policy_optimizer = optimizers.Adam().setup(policy)
        policy_optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(1))

        def make_q_func_with_optimizer():
            q_func = chainer.Sequential(
                concat_obs_and_action,
                L.Linear(obs_size + action_size, hidden_size),
                F.relu,
                L.Linear(hidden_size, 1,
                         initialW=chainer.initializers.LeCunNormal(1e-1)),
            )
            q_func_optimizer = optimizers.Adam(1e-2).setup(q_func)
            q_func_optimizer.add_hook(
                chainer.optimizer_hooks.GradientClipping(1))
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        rbuf = chainerrl.replay_buffer.ReplayBuffer(10 ** 6)

        explorer = chainerrl.explorers.AdditiveGaussian(
            scale=0.3, low=env.action_space.low, high=env.action_space.high)

        def burnin_action_func():
            return np.random.uniform(
                env.action_space.low, env.action_space.high).astype(np.float32)

        agent = chainerrl.agents.TD3(
            policy=policy,
            q_func1=q_func1,
            q_func2=q_func2,
            policy_optimizer=policy_optimizer,
            q_func1_optimizer=q_func1_optimizer,
            q_func2_optimizer=q_func2_optimizer,
            replay_buffer=rbuf,
            explorer=explorer,
            gamma=0.5,
            minibatch_size=100,
            replay_start_size=100,
            burnin_action_func=burnin_action_func,
        )

        return agent

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=False,
            episodic=self.episodic or test,
            deterministic=test,
        )
        return env, 1

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [make_env for _ in range(num_envs)])
        return vec_env, 1.0
