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
from chainerrl.agents import trpo
from chainerrl.envs.abc import ABC
from chainerrl.experiments.evaluator import batch_run_evaluation_episodes
from chainerrl.experiments.evaluator import run_evaluation_episodes
from chainerrl.experiments import train_agent_batch_with_evaluation
from chainerrl.experiments import train_agent_with_evaluation
from chainerrl.links import StatelessRecurrentSequential
from chainerrl import policies


class OldStyleIdentity(chainer.Function):

    """Old-style identity function."""

    def check_type_forward(self, in_types):
        pass

    def forward(self, xs):
        self.retain_inputs(())
        return xs

    def backward(self, xs, gys):
        return gys


def old_style_identity(*args):
    return OldStyleIdentity()(*args)


class TestFindOldStyleFunction(unittest.TestCase):

    def test(self):
        a = chainer.Variable(np.random.rand(1).astype(np.float32))
        b = chainer.Variable(np.random.rand(1).astype(np.float32))

        # No old-style function
        y = 2 * a + b
        old_style_funcs = trpo._find_old_style_function([y])
        self.assertEqual(old_style_funcs, [])

        # One old-style function
        y = 2 * old_style_identity(a) + b
        old_style_funcs = trpo._find_old_style_function([y])
        self.assertEqual(len(old_style_funcs), 1)
        self.assertTrue(all(isinstance(f, OldStyleIdentity)
                            for f in old_style_funcs))

        # Three old-style functions
        y = (2 * old_style_identity(old_style_identity(a))
             + old_style_identity(b))
        old_style_funcs = trpo._find_old_style_function([y])
        self.assertEqual(len(old_style_funcs), 3)
        self.assertTrue(all(isinstance(f, OldStyleIdentity)
                            for f in old_style_funcs))


def compute_hessian_vector_product(y, params, vec):
    grads = chainer.grad(
        [y], params, enable_double_backprop=True)
    flat_grads = trpo._flatten_and_concat_variables(grads)
    return trpo._hessian_vector_product(flat_grads, params, vec)


def compute_hessian(y, params):
    grads = chainer.grad(
        [y], params, enable_double_backprop=True)
    flat_grads = trpo._flatten_and_concat_variables(grads)
    hessian_rows = []
    for i in range(len(flat_grads)):
        ggrads = chainer.grad([flat_grads[i]], params)
        assert all(ggrad is not None for ggrad in ggrads)
        ggrads_data = [ggrad.array for ggrad in ggrads]
        flat_ggrads_data = trpo._flatten_and_concat_ndarrays(ggrads_data)
        hessian_rows.append(flat_ggrads_data)
    return np.asarray(hessian_rows)


class TestHessianVectorProduct(unittest.TestCase):

    def _generate_params_and_first_order_output(self):
        a = chainer.Variable(np.random.rand(3).astype(np.float32))
        b = chainer.Variable(np.random.rand(1).astype(np.float32))
        params = [a, b]
        y = F.sum(a, keepdims=True) * 3 + b
        return params, y

    def _generate_params_and_second_order_output(self):
        a = chainer.Variable(np.random.rand(3).astype(np.float32))
        b = chainer.Variable(np.random.rand(1).astype(np.float32))
        params = [a, b]
        y = F.sum(a, keepdims=True) * 3 * b
        return params, y

    def test_first_order(self):
        # First order, so its Hessian will contain None
        params, y = self._generate_params_and_first_order_output()

        old_style_funcs = trpo._find_old_style_function([y])
        if old_style_funcs:
            self.skipTest("\
Chainer v{} does not support double backprop of these functions: {}.".format(
                chainer.__version__, old_style_funcs))

        vec = np.random.rand(4).astype(np.float32)
        # Hessian-vector product computation should raise an error due to None
        with self.assertRaises(AssertionError):
            compute_hessian_vector_product(y, params, vec)

    def test_second_order(self):
        # Second order, so its Hessian will be non-zero
        params, y = self._generate_params_and_second_order_output()

        old_style_funcs = trpo._find_old_style_function([y])
        if old_style_funcs:
            self.skipTest("\
Chainer v{} does not support double backprop of these functions: {}.".format(
                chainer.__version__, old_style_funcs))

        def test_hessian_vector_product_nonzero(vec):
            hvp = compute_hessian_vector_product(y, params, vec)
            hessian = compute_hessian(y, params)
            self.assertGreater(np.count_nonzero(hvp), 0)
            self.assertGreater(np.count_nonzero(hessian), 0)
            np.testing.assert_allclose(hvp, hessian.dot(vec), atol=1e-3)

        # Test with two different random vectors, reusing y
        test_hessian_vector_product_nonzero(
            np.random.rand(4).astype(np.float32))
        test_hessian_vector_product_nonzero(
            np.random.rand(4).astype(np.float32))


@testing.parameterize(*(
    testing.product({
        'discrete': [False, True],
        'episodic': [False, True],
        'lambd': [0.0, 0.5, 1.0],
        'entropy_coef': [0.0, 1e-5],
        'standardize_advantages': [False, True],
        'standardize_obs': [False, True],
        'recurrent': [False],
    })
    +
    testing.product({
        'discrete': [False, True],
        'episodic': [False, True],
        'lambd': [0.9],
        'entropy_coef': [1e-5],
        'standardize_advantages': [True],
        'standardize_obs': [True],
        'recurrent': [True],
    })
))
class TestTRPO(unittest.TestCase):

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

        if self.recurrent and gpu >= 0:
            self.skipTest(
                'NStepLSTM does not support double backprop with GPU.')
        if self.recurrent and chainer.__version__ == '7.0.0b3':
            self.skipTest(
                'chainer==7.0.0b3 has a bug in double backrop of LSTM.'
                ' See https://github.com/chainer/chainer/pull/8037')

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

        if self.recurrent and gpu >= 0:
            self.skipTest(
                'NStepLSTM does not support double backprop with GPU.')
        if self.recurrent and chainer.__version__ == '7.0.0b3':
            self.skipTest(
                'chainer==7.0.0b3 has a bug in double backrop of LSTM.'
                ' See https://github.com/chainer/chainer/pull/8037')

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
        policy, vf = self.make_model(env)

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            policy.to_gpu(gpu)
            vf.to_gpu(gpu)

        vf_opt = optimizers.Adam(alpha=1e-2)
        vf_opt.setup(vf)
        vf_opt.add_hook(chainer.optimizer_hooks.GradientClipping(1))

        if self.standardize_obs:
            obs_normalizer = chainerrl.links.EmpiricalNormalization(
                env.observation_space.low.size)
            if gpu >= 0:
                obs_normalizer.to_gpu(gpu)
        else:
            obs_normalizer = None

        agent = chainerrl.agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            obs_normalizer=obs_normalizer,
            gamma=0.5,
            lambd=self.lambd,
            entropy_coef=self.entropy_coef,
            standardize_advantages=self.standardize_advantages,
            update_interval=64,
            vf_batch_size=32,
            act_deterministically=True,
            recurrent=self.recurrent,
        )

        return agent

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
        return pi, v

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=self.discrete,
            episodic=self.episodic or test,
            deterministic=test,
            partially_observable=self.recurrent,
        )
        return env, 1

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [make_env for _ in range(num_envs)])
        return vec_env, 1.0
