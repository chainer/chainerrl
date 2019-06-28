from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
import itertools
import os
import tempfile
import unittest

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import testing
import numpy as np

import chainerrl
from chainerrl.agents.a3c import A3CSeparateModel
from chainerrl.agents.ppo import PPO
from chainerrl.agents import ppo
from chainerrl.envs.abc import ABC
from chainerrl.experiments.evaluator import batch_run_evaluation_episodes
from chainerrl.experiments.evaluator import run_evaluation_episodes
from chainerrl.experiments import train_agent_batch_with_evaluation
from chainerrl.experiments import train_agent_with_evaluation
from chainerrl.misc.batch_states import batch_states
from chainerrl import policies

from chainerrl.links import StatelessRecurrentSequential
from chainerrl.links import StatelessRecurrentBranched


def make_random_episodes(n_episodes=10, obs_size=2, n_actions=3):
    episodes = []
    for _ in range(n_episodes):
        episode_length = np.random.randint(1, 100)
        episode = []
        last_state = np.random.uniform(-1, 1, size=obs_size)
        for t in range(episode_length):
            state = np.random.uniform(-1, 1, size=obs_size)
            episode.append({
                'state': last_state,
                'action': np.random.randint(n_actions),
                'reward': np.random.uniform(-1, 1),
                'nonterminal': (np.random.randint(2)
                                if t == episode_length - 1 else 1),
                'next_state': state,
                'recurrent_state': None,
                'next_recurrent_state': None,
            })
            last_state = state
        episodes.append(episode)

    assert len(episodes) == n_episodes
    return episodes


class TestYieldSubsetOfSequencesWithFixedNumberOfItems(unittest.TestCase):

    def test_manual(self):
        episodes = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8],
            [9],
            [10, 11, 12],
        ]
        self.assertEqual(
            list(ppo._yield_subset_of_sequences_with_fixed_number_of_items(
                episodes, 4)),
            [
                [[1, 2, 3], [4]],
                [[5], [6, 7, 8]],
                [[9], [10, 11, 12]],
            ],
        )
        self.assertEqual(
            list(ppo._yield_subset_of_sequences_with_fixed_number_of_items(
                episodes, 3)),
            [
                [[1, 2, 3]],
                [[4, 5], [6]],
                [[7, 8], [9]],
                [[10, 11, 12]],
            ],
        )
        self.assertEqual(
            list(ppo._yield_subset_of_sequences_with_fixed_number_of_items(
                episodes, 2)),
            [
                [[1, 2]],
                [[3], [4]],
                [[5], [6]],
                [[7, 8]],
                [[9], [10]],
                [[11, 12]],
            ],
        )


class TestLimitSequenceLength(unittest.TestCase):

    def test_manual(self):
        episodes = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8],
            [9],
        ]
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 1),
            [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 2),
            [
                [1, 2],
                [3],
                [4, 5],
                [6, 7],
                [8],
                [9],
            ],
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 3),
            episodes,
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 4),
            episodes,
        )

    def test_random(self):
        episodes = make_random_episodes()
        limit = 5
        new_episodes = chainerrl.agents.ppo._limit_sequence_length(
            episodes, limit)
        for ep in new_episodes:
            self.assertLessEqual(len(ep), limit)
        # They should have the same number of transitions
        self.assertEqual(
            sum(len(ep) for ep in episodes),
            sum(len(ep) for ep in new_episodes))


@testing.parameterize(*(
    testing.product({
        'use_obs_normalizer': [True, False],
        'gamma': [1, 0.8, 0],
        'lambd': [1, 0.8, 0],
        'max_recurrent_sequence_len': [None, 7],
    })
))
class TestPPODataset(unittest.TestCase):

    def test_recurrent_and_non_recurrent_equivalence(self):
        """Test equivalence between recurrent and non-recurrent datasets.

        When the same feed-forward model is used, the values of
        log_prob, v_pred, next_v_pred obtained by both recurrent and
        non-recurrent dataset creation functions should be the same.
        """
        episodes = make_random_episodes()
        if self.use_obs_normalizer:
            obs_normalizer = chainerrl.links.EmpiricalNormalization(
                2, clip_threshold=5)
            obs_normalizer.experience(
                np.random.uniform(-1, 1, size=(10, 2)))
        else:
            obs_normalizer = None

        def phi(obs):
            return (obs * 0.5).astype(np.float32)

        obs_size = 2
        n_actions = 3

        non_recurrent_model = A3CSeparateModel(
            pi=chainerrl.policies.FCSoftmaxPolicy(obs_size, n_actions),
            v=L.Linear(obs_size, 1),
        )
        recurrent_model = StatelessRecurrentSequential(
            non_recurrent_model,
        )
        xp = non_recurrent_model.xp

        dataset = chainerrl.agents.ppo._make_dataset(
            episodes=copy.deepcopy(episodes),
            model=non_recurrent_model,
            phi=phi,
            batch_states=batch_states,
            obs_normalizer=obs_normalizer,
            gamma=self.gamma,
            lambd=self.lambd,
        )

        dataset_recurrent = chainerrl.agents.ppo._make_dataset_recurrent(
            episodes=copy.deepcopy(episodes),
            model=recurrent_model,
            phi=phi,
            batch_states=batch_states,
            obs_normalizer=obs_normalizer,
            gamma=self.gamma,
            lambd=self.lambd,
            max_recurrent_sequence_len=self.max_recurrent_sequence_len,
        )

        self.assertTrue('log_prob' not in episodes[0][0])
        self.assertTrue('log_prob' in dataset[0])
        self.assertTrue('log_prob' in dataset_recurrent[0][0])
        # They are not just shallow copies
        self.assertTrue(dataset[0]['log_prob']
                        is not dataset_recurrent[0][0]['log_prob'])

        states = [tr['state'] for tr in dataset]
        recurrent_states = [
            tr['state'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(states, recurrent_states)

        actions = [tr['action'] for tr in dataset]
        recurrent_actions = [
            tr['action'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(actions, recurrent_actions)

        rewards = [tr['reward'] for tr in dataset]
        recurrent_rewards = [
            tr['reward'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(rewards, recurrent_rewards)

        nonterminals = [tr['nonterminal'] for tr in dataset]
        recurrent_nonterminals = [
            tr['nonterminal'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(nonterminals, recurrent_nonterminals)

        log_probs = [tr['log_prob'] for tr in dataset]
        recurrent_log_probs = [
            tr['log_prob'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(log_probs, recurrent_log_probs)

        vs_pred = [tr['v_pred'] for tr in dataset]
        recurrent_vs_pred = [
            tr['v_pred'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(vs_pred, recurrent_vs_pred)

        next_vs_pred = [tr['next_v_pred'] for tr in dataset]
        recurrent_next_vs_pred = [
            tr['next_v_pred'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(next_vs_pred, recurrent_next_vs_pred)

        advs = [tr['adv'] for tr in dataset]
        recurrent_advs = [
            tr['adv'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(advs, recurrent_advs)

        vs_teacher = [tr['v_teacher'] for tr in dataset]
        recurrent_vs_teacher = [
            tr['v_teacher'] for tr in itertools.chain.from_iterable(
                dataset_recurrent)]
        xp.testing.assert_allclose(vs_teacher, recurrent_vs_teacher)


@testing.parameterize(*(
    testing.product({
        'clip_eps_vf': [None, 0.2],
        'lambd': [0.0, 0.5],
        'discrete': [False, True],
        'standardize_advantages': [False, True],
        'episodic': [True, False],
        'recurrent': [False],
    })
    +
    testing.product({
        'clip_eps_vf': [0.2],
        'lambd': [.0, 0.5],
        'discrete': [False, True],
        'standardize_advantages': [True],
        'episodic': [True, False],
        'recurrent': [True],
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
        max_episode_len = None if self.episodic else 2

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=50,
            successful_score=successful_return,
            eval_env=test_env,
            train_max_episode_len=max_episode_len,
        )

        agent.stop_episode()

        # Test
        n_test_runs = 10
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

    def _test_abc_batch(
            self, steps=100000,
            require_success=True, gpu=-1, load_model=False, num_envs=4):

        env, _ = self.make_vec_env_and_successful_return(
            test=False, num_envs=num_envs)
        test_env, successful_return = self.make_vec_env_and_successful_return(
            test=True, num_envs=num_envs)
        agent = self.make_agent(env, gpu)
        max_episode_len = None if self.episodic else 2

        if load_model:
            print('Load agent from', self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=40,
            successful_score=successful_return,
            eval_env=test_env,
            log_interval=100,
            max_episode_len=max_episode_len,
        )
        env.close()

        # Test
        n_test_runs = 10
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
        model = self.make_model(env)

        opt = optimizers.Adam(alpha=1e-2)
        opt.setup(model)
        opt.add_hook(chainer.optimizer_hooks.GradientClipping(1))

        return self.make_ppo_agent(env=env, model=model, opt=opt, gpu=gpu)

    def make_ppo_agent(self, env, model, opt, gpu):
        return PPO(
            model, opt, gpu=gpu, gamma=0.8, lambd=self.lambd,
            update_interval=64, minibatch_size=16, epochs=3,
            clip_eps_vf=self.clip_eps_vf,
            standardize_advantages=self.standardize_advantages,
            recurrent=self.recurrent,
            entropy_coef=1e-5,
            act_deterministically=True,
        )

    def make_model(self, env):
        n_hidden_channels = 20
        obs_size = env.observation_space.low.size

        if self.recurrent:
            v = StatelessRecurrentSequential(
                L.NStepLSTM(1, obs_size, n_hidden_channels, 0),
                L.Linear(
                    None, 1, initialW=chainer.initializers.LeCunNormal(1e-1)),
            )
            if self.discrete:
                n_actions = env.action_space.n
                pi = StatelessRecurrentSequential(
                    L.NStepLSTM(1, obs_size, n_hidden_channels, 0),
                    policies.FCSoftmaxPolicy(
                        n_hidden_channels, n_actions,
                        n_hidden_layers=0,
                        nonlinearity=F.tanh,
                        last_wscale=1e-1,
                    )
                )
            else:
                action_size = env.action_space.low.size
                pi = StatelessRecurrentSequential(
                    L.NStepLSTM(1, obs_size, n_hidden_channels, 0),
                    policies.FCGaussianPolicy(
                        n_hidden_channels, action_size,
                        n_hidden_layers=0,
                        nonlinearity=F.tanh,
                        mean_wscale=1e-1,
                    )
                )
            return StatelessRecurrentBranched(pi, v)
        else:
            v = chainer.Sequential(
                L.Linear(None, n_hidden_channels),
                F.tanh,
                L.Linear(
                    None, 1, initialW=chainer.initializers.LeCunNormal(1e-1)),
            )
            if self.discrete:
                n_actions = env.action_space.n
                pi = policies.FCSoftmaxPolicy(
                    obs_size, n_actions,
                    n_hidden_layers=1,
                    n_hidden_channels=n_hidden_channels,
                    nonlinearity=F.tanh,
                    last_wscale=1e-1,
                )
            else:
                action_size = env.action_space.low.size
                pi = policies.FCGaussianPolicy(
                    obs_size, action_size,
                    n_hidden_layers=1,
                    n_hidden_channels=n_hidden_channels,
                    nonlinearity=F.tanh,
                    mean_wscale=1e-1,
                )
            return A3CSeparateModel(pi=pi, v=v)

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=self.discrete,
            deterministic=test,
            episodic=self.episodic,
            partially_observable=self.recurrent,
        )
        return env, 1.0

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [make_env for _ in range(num_envs)])
        return vec_env, 1.0
