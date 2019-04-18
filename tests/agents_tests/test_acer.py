from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
import logging
import os
import tempfile
import unittest
import warnings

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import testing
from chainer.testing import condition
import numpy as np

import chainerrl
from chainerrl.agents import acer
from chainerrl.envs.abc import ABC
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import q_function
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl import v_function


def extract_gradients_as_single_vector(link):
    return np.concatenate([p.grad.ravel() for p in link.params()])


@testing.parameterize(
    *testing.product({
        'distrib_type': ['Gaussian', 'Softmax'],
        'pi_deg': [True, False],
        'mu_deg': [True, False],
        'truncation_threshold': [0, 1, 10, None],
    })
)
class TestDegenerateDistribution(unittest.TestCase):

    def setUp(self):
        if self.distrib_type == 'Gaussian':
            action_size = 2
            W = np.random.rand(action_size, 1).astype(np.float32)
            self.action_value = chainerrl.action_value.SingleActionValue(
                evaluator=lambda x: chainer.Variable(
                    np.asarray(np.dot(x, W), dtype=np.float32)))
            nondeg_distrib = chainerrl.distribution.GaussianDistribution(
                mean=np.random.rand(1, action_size).astype(np.float32),
                var=np.full((1, action_size), 1, dtype=np.float32))
            deg_distrib = chainerrl.distribution.GaussianDistribution(
                mean=np.random.rand(1, action_size).astype(np.float32),
                var=np.full((1, action_size), 1e-10, dtype=np.float32))
        elif self.distrib_type == 'Softmax':
            q_values = chainer.Variable(
                np.asarray([[1, 3]], dtype=np.float32))
            self.action_value = chainerrl.action_value.DiscreteActionValue(
                q_values)
            nondeg_logits = chainer.Variable(
                np.asarray([[0, 0]], dtype=np.float32))
            nondeg_distrib = chainerrl.distribution.SoftmaxDistribution(
                nondeg_logits)
            deg_logits = chainer.Variable(
                np.asarray([[1e10, 1e-10]], dtype=np.float32))
            deg_distrib = chainerrl.distribution.SoftmaxDistribution(
                deg_logits)
        self.pi = deg_distrib if self.pi_deg else nondeg_distrib
        self.mu = deg_distrib if self.mu_deg else nondeg_distrib

    def test_importance(self):
        action = self.mu.sample().array
        pimu = acer.compute_importance(self.pi, self.mu, action)
        print('pi/mu', pimu)
        self.assertFalse(np.isnan(np.sum(pimu)))

    def test_full_importance(self):
        if self.distrib_type == 'Gaussian':
            return
        pimu = acer.compute_full_importance(self.pi, self.mu)
        print('pi/mu', pimu)
        self.assertFalse(np.isnan(np.sum(pimu)))

    def test_full_correction_term(self):
        if self.distrib_type == 'Gaussian':
            return
        if self.truncation_threshold is None:
            return
        correction_term = acer.compute_policy_gradient_full_correction(
            self.pi, self.mu, self.action_value, 0, self.truncation_threshold)
        print('correction_term', correction_term.array)
        self.assertFalse(np.isnan(np.sum(correction_term.array)))

    def test_sample_correction_term(self):
        if self.truncation_threshold is None:
            return
        correction_term = acer.compute_policy_gradient_sample_correction(
            self.pi, self.mu, self.action_value, 0, self.truncation_threshold)
        print('correction_term', correction_term.array)
        self.assertFalse(np.isnan(np.sum(correction_term.array)))

    def test_policy_gradient(self):
        action = self.mu.sample().array
        pg = acer.compute_policy_gradient_loss(
            action, 1, self.pi, self.mu, self.action_value, 0,
            self.truncation_threshold)
        print('pg', pg.array)
        self.assertFalse(np.isnan(np.sum(pg.array)))


@testing.parameterize(*(
    testing.product({
        'distrib_type': ['Gaussian'],
        'action_size': [1, 2]
    }) +
    testing.product({
        'distrib_type': ['Softmax'],
        'n_actions': [2, 3]
    })
))
class TestBiasCorrection(unittest.TestCase):

    def setUp(self):
        pass

    @testing.attr.slow
    @condition.retry(3)
    def test_bias_correction(self):

        if self.distrib_type == 'Gaussian':
            base_policy = chainerrl.policies.FCGaussianPolicy(
                1, self.action_size, n_hidden_channels=0, n_hidden_layers=0)
            another_policy = chainerrl.policies.FCGaussianPolicy(
                1, self.action_size, n_hidden_channels=0, n_hidden_layers=0)
        elif self.distrib_type == 'Softmax':
            base_policy = chainerrl.policies.FCSoftmaxPolicy(
                1, self.n_actions, n_hidden_channels=0, n_hidden_layers=0)
            another_policy = chainerrl.policies.FCSoftmaxPolicy(
                1, self.n_actions, n_hidden_channels=0, n_hidden_layers=0)
        x = np.full((1, 1), 1, dtype=np.float32)
        pi = base_policy(x)
        mu = another_policy(x)

        def evaluate_action(action):
            return float(action_value.evaluate_actions(action).array)

        if self.distrib_type == 'Gaussian':
            W = np.random.rand(self.action_size, 1).astype(np.float32)
            action_value = chainerrl.action_value.SingleActionValue(
                evaluator=lambda x: chainer.Variable(
                    np.asarray(np.dot(x, W), dtype=np.float32)))
        else:
            q_values = np.zeros((1, self.n_actions), dtype=np.float32)
            q_values[:, np.random.randint(self.n_actions)] = 1
            action_value = chainerrl.action_value.DiscreteActionValue(
                chainer.Variable(q_values))

        n = 1000

        pi_samples = [pi.sample().array for _ in range(n)]
        mu_samples = [mu.sample().array for _ in range(n)]

        onpolicy_gs = []
        for sample in pi_samples:
            base_policy.cleargrads()
            loss = -evaluate_action(sample) * pi.log_prob(sample)
            F.squeeze(loss).backward()
            onpolicy_gs.append(extract_gradients_as_single_vector(base_policy))
        # on-policy
        onpolicy_gs_mean = np.mean(onpolicy_gs, axis=0)
        onpolicy_gs_var = np.var(onpolicy_gs, axis=0)
        print('on-policy')
        print('g mean', onpolicy_gs_mean)
        print('g var', onpolicy_gs_var)

        # off-policy without importance sampling
        offpolicy_gs = []
        for sample in mu_samples:
            base_policy.cleargrads()
            loss = -evaluate_action(sample) * pi.log_prob(sample)
            F.squeeze(loss).backward()
            offpolicy_gs.append(
                extract_gradients_as_single_vector(base_policy))
        offpolicy_gs_mean = np.mean(offpolicy_gs, axis=0)
        offpolicy_gs_var = np.var(offpolicy_gs, axis=0)
        print('off-policy')
        print('g mean', offpolicy_gs_mean)
        print('g var', offpolicy_gs_var)

        # off-policy with importance sampling
        is_gs = []
        for sample in mu_samples:
            base_policy.cleargrads()
            rho = float(pi.prob(sample).array / mu.prob(sample).array)
            loss = -rho * evaluate_action(sample) * pi.log_prob(sample)
            F.squeeze(loss).backward()
            is_gs.append(extract_gradients_as_single_vector(base_policy))
        is_gs_mean = np.mean(is_gs, axis=0)
        is_gs_var = np.var(is_gs, axis=0)
        print('importance sampling')
        print('g mean', is_gs_mean)
        print('g var', is_gs_var)

        # off-policy with truncated importance sampling + bias correction
        def bias_correction_policy_gradients(truncation_threshold):
            gs = []
            for sample in mu_samples:
                base_policy.cleargrads()
                loss = acer.compute_policy_gradient_loss(
                    action=sample,
                    advantage=evaluate_action(sample),
                    action_distrib=pi,
                    action_distrib_mu=mu,
                    action_value=action_value,
                    v=0,
                    truncation_threshold=truncation_threshold)
                F.squeeze(loss).backward()
                gs.append(extract_gradients_as_single_vector(base_policy))
            return gs

        # c=0 means on-policy sampling
        print('truncated importance sampling + bias correction c=0')
        tis_c0_gs = bias_correction_policy_gradients(0)
        tis_c0_gs_mean = np.mean(tis_c0_gs, axis=0)
        tis_c0_gs_var = np.var(tis_c0_gs, axis=0)
        print('g mean', tis_c0_gs_mean)
        print('g var', tis_c0_gs_var)
        # c=0 must be low-bias compared to naive off-policy sampling
        self.assertLessEqual(
            np.linalg.norm(onpolicy_gs_mean - tis_c0_gs_mean),
            np.linalg.norm(onpolicy_gs_mean - offpolicy_gs_mean))

        # c=1 means truncated importance sampling with bias correction
        print('truncated importance sampling + bias correction c=1')
        tis_c1_gs = bias_correction_policy_gradients(1)
        tis_c1_gs_mean = np.mean(tis_c1_gs, axis=0)
        tis_c1_gs_var = np.var(tis_c1_gs, axis=0)
        print('g mean', tis_c1_gs_mean)
        print('g var', tis_c1_gs_var)
        # c=1 must be low-variance compared to naive importance sampling
        self.assertLessEqual(tis_c1_gs_var.sum(), is_gs_var.sum())
        # c=1 must be low-bias compared to naive off-policy sampling
        self.assertLess(
            np.linalg.norm(onpolicy_gs_mean - tis_c1_gs_mean),
            np.linalg.norm(onpolicy_gs_mean - offpolicy_gs_mean))

        # c=inf means importance sampling no truncation
        print('truncated importance sampling + bias correction c=inf')
        tis_cinf_gs = bias_correction_policy_gradients(np.inf)
        tis_cinf_gs_mean = np.mean(tis_cinf_gs, axis=0)
        tis_cinf_gs_var = np.var(tis_cinf_gs, axis=0)
        print('g mean', tis_cinf_gs_mean)
        print('g var', tis_cinf_gs_var)
        np.testing.assert_allclose(tis_cinf_gs_mean, is_gs_mean, rtol=1e-3)
        np.testing.assert_allclose(tis_cinf_gs_var, is_gs_var, rtol=1e-3)


@testing.parameterize(
    *testing.product({
        'distrib_type': ['Gaussian', 'Softmax'],
    })
)
class TestEfficientTRPO(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_loss_with_kl_constraint(self):

        if self.distrib_type == 'Gaussian':
            base_policy = chainerrl.policies.FCGaussianPolicy(
                1, 3, n_hidden_channels=0, n_hidden_layers=0)
            x = np.random.rand(1, 1).astype(np.float32, copy=False)
        elif self.distrib_type == 'Softmax':
            base_policy = chainerrl.policies.FCSoftmaxPolicy(
                1, 3, n_hidden_channels=0, n_hidden_layers=0)
            x = np.random.rand(1, 1).astype(np.float32, copy=False)
        base_distrib = base_policy(x)
        sample = base_distrib.sample().array
        another_distrib = base_policy(
            np.random.rand(1, 1).astype(np.float32, copy=False)).copy()

        def base_loss_func(distrib):
            return distrib.log_prob(sample)

        kl_before = float(another_distrib.kl(base_distrib).array)
        print('kl_before', kl_before)

        def compute_kl_after_update(loss_func, n=100):
            policy = copy.deepcopy(base_policy)
            optimizer = chainer.optimizers.SGD(1e-4)
            optimizer.setup(policy)
            for _ in range(n):
                distrib = policy(x)
                policy.cleargrads()
                F.squeeze(loss_func(distrib)).backward()
                optimizer.update()
            distrib_after = policy(x)
            return float(another_distrib.kl(distrib_after).array)

        # Without kl constraint
        kl_after_without_constraint = compute_kl_after_update(base_loss_func)
        print('kl_after_without_constraint', kl_after_without_constraint)

        # With kl constraint
        def loss_func_with_constraint(distrib):
            loss, kl = acer.compute_loss_with_kl_constraint(
                distrib, another_distrib, base_loss_func(distrib),
                delta=0)
            return loss
        kl_after_with_constraint = compute_kl_after_update(
            loss_func_with_constraint)
        print('kl_after_with_constraint', kl_after_with_constraint)

        # TODO(fujita) check the results are correct


@testing.parameterize(*(
    testing.product({
        'discrete': [True, False],
        't_max': [1, 2],
        'use_lstm': [False],
        'episodic': [True, False],
        'n_times_replay': [0, 2],
        'disable_online_update': [True, False],
        'use_trust_region': [True, False],
    }) +
    testing.product({
        'discrete': [True, False],
        't_max': [5],
        'use_lstm': [True],
        'episodic': [True, False],
        'n_times_replay': [0, 2],
        'disable_online_update': [True, False],
        'use_trust_region': [True, False],
    })
))
class TestACER(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @testing.attr.slow
    def test_abc(self):
        self._test_abc(self.t_max, self.use_lstm, discrete=self.discrete,
                       episodic=self.episodic)

    def test_abc_fast(self):
        self._test_abc(self.t_max, self.use_lstm, discrete=self.discrete,
                       episodic=self.episodic, steps=10, require_success=False)

    def _test_abc(self, t_max, use_lstm, discrete=True, episodic=True,
                  steps=100000, require_success=True):

        nproc = 8

        def make_env(process_idx, test):
            size = 2
            return ABC(size=size, discrete=discrete, episodic=episodic or test,
                       partially_observable=self.use_lstm,
                       deterministic=test)

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        def phi(x):
            return x

        n_hidden_channels = 20
        n_hidden_layers = 1
        nonlinearity = F.leaky_relu
        replay_buffer = EpisodicReplayBuffer(10 ** 4)
        if use_lstm:
            if discrete:
                model = acer.ACERSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCSoftmaxPolicy(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        min_prob=1e-1),
                    q=q_function.FCStateQFunctionWithDiscreteAction(
                        n_hidden_channels, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                )
            else:
                model = acer.ACERSDNSharedModel(
                    shared=L.LSTM(obs_space.low.size, n_hidden_channels),
                    pi=policies.FCGaussianPolicy(
                        n_hidden_channels, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high,
                        nonlinearity=nonlinearity,
                        min_var=1e-1),
                    v=v_function.FCVFunction(
                        n_hidden_channels,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                    adv=q_function.FCSAQFunction(
                        n_hidden_channels, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                )
        else:
            if discrete:
                model = acer.ACERSeparateModel(
                    pi=policies.FCSoftmaxPolicy(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity,
                        min_prob=1e-1),
                    q=q_function.FCStateQFunctionWithDiscreteAction(
                        obs_space.low.size, action_space.n,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                )
            else:
                model = acer.ACERSDNSeparateModel(
                    pi=policies.FCGaussianPolicy(
                        obs_space.low.size, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        bound_mean=True,
                        min_action=action_space.low,
                        max_action=action_space.high,
                        nonlinearity=nonlinearity,
                        min_var=1e-1),
                    v=v_function.FCVFunction(
                        obs_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                    adv=q_function.FCSAQFunction(
                        obs_space.low.size, action_space.low.size,
                        n_hidden_channels=n_hidden_channels,
                        n_hidden_layers=n_hidden_layers,
                        nonlinearity=nonlinearity),
                )
        eps = 1e-8
        opt = rmsprop_async.RMSpropAsync(lr=1e-3, eps=eps, alpha=0.99)
        opt.setup(model)
        gamma = 0.5
        beta = 1e-5
        if self.n_times_replay == 0 and self.disable_online_update:
            # At least one of them must be enabled
            return
        agent = acer.ACER(
            model, opt, replay_buffer=replay_buffer,
            t_max=t_max, gamma=gamma, beta=beta,
            phi=phi,
            n_times_replay=self.n_times_replay,
            act_deterministically=True,
            disable_online_update=self.disable_online_update,
            replay_start_size=100,
            use_trust_region=self.use_trust_region)

        max_episode_len = None if episodic else 2

        with warnings.catch_warnings(record=True) as warns:
            train_agent_async(
                outdir=self.outdir, processes=nproc, make_env=make_env,
                agent=agent, steps=steps,
                max_episode_len=max_episode_len,
                eval_interval=500,
                eval_n_steps=None,
                eval_n_episodes=5,
                successful_score=1)
            assert len(warns) == 0, warns[0]

        # The agent returned by train_agent_async is not guaranteed to be
        # successful because parameters could be modified by other processes
        # after success. Thus here the successful model is loaded explicitly.
        if require_success:
            agent.load(os.path.join(self.outdir, 'successful'))
        agent.stop_episode()

        # Test
        env = make_env(0, True)
        n_test_runs = 5

        for _ in range(n_test_runs):
            total_r = 0
            obs = env.reset()
            done = False
            reward = 0.0

            while not done:
                action = agent.act(obs)
                print('state:', obs, 'action:', action)
                obs, reward, done, _ = env.step(action)
                total_r += reward
            if require_success:
                self.assertAlmostEqual(total_r, 1)
            agent.stop_episode()
