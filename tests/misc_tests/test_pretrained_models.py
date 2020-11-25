import functools
import os
import unittest

import chainer
import chainer.functions as F
from chainer import links as L
from chainer import optimizers
from chainer import testing
import numpy as np

import chainerrl
from chainerrl import agents
from chainerrl import explorers
from chainerrl import links
from chainerrl import policies
from chainerrl.optimizers import rmsprop_async
from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_functions import DistributionalDuelingDQN
from chainerrl.misc import download_model
from chainerrl import replay_buffer
from chainerrl import v_functions


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadDQN(unittest.TestCase):

    def _test_load_dqn(self, gpu):
        q_func = links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, 4),
            DiscreteActionValue)

        opt = optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)
        opt.setup(q_func)

        rbuf = replay_buffer.ReplayBuffer(100)

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(4))

        agent = agents.DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.99,
                           explorer=explorer, replay_start_size=50,
                           target_update_interval=10 ** 4,
                           clip_delta=True,
                           update_interval=4,
                           batch_accumulator='sum',
                           phi=lambda x: x)

        model, exists = download_model("DQN", "BreakoutNoFrameskip-v4",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_dqn(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_dqn(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadIQN(unittest.TestCase):

    def _test_load_iqn(self, gpu):
        q_func = agents.iqn.ImplicitQuantileQFunction(
            psi=links.Sequence(
                L.Convolution2D(None, 32, 8, stride=4),
                F.relu,
                L.Convolution2D(None, 64, 4, stride=2),
                F.relu,
                L.Convolution2D(None, 64, 3, stride=1),
                F.relu,
                functools.partial(F.reshape, shape=(-1, 3136)),
            ),
            phi=links.Sequence(
                agents.iqn.CosineBasisLinear(64, 3136),
                F.relu,
            ),
            f=links.Sequence(
                L.Linear(None, 512),
                F.relu,
                L.Linear(None, 4),
            ),)

        opt = chainer.optimizers.Adam(5e-5, eps=1e-2)
        opt.setup(q_func)

        rbuf = replay_buffer.ReplayBuffer(100)

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(4))

        agent = agents.IQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.99,
            explorer=explorer, replay_start_size=50,
            target_update_interval=10 ** 4,
            update_interval=4,
            batch_accumulator='mean',
            phi=lambda x: x,
            quantile_thresholds_N=64,
            quantile_thresholds_N_prime=64,
            quantile_thresholds_K=32,
        )

        model, exists = download_model("IQN", "BreakoutNoFrameskip-v4",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_iqn(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_iqn(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadRainbow(unittest.TestCase):

    def _test_load_rainbow(self, gpu):
        q_func = DistributionalDuelingDQN(4, 51, -10, 10)
        links.to_factorized_noisy(q_func, sigma_scale=0.5)
        explorer = explorers.Greedy()
        opt = chainer.optimizers.Adam(6.25e-5, eps=1.5 * 10 ** -4)
        opt.setup(q_func)
        rbuf = replay_buffer.ReplayBuffer(100)
        agent = agents.CategoricalDoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.99,
            explorer=explorer, minibatch_size=32,
            replay_start_size=50,
            target_update_interval=32000,
            update_interval=4,
            batch_accumulator='mean',
            phi=lambda x: x,
        )

        model, exists = download_model("Rainbow", "BreakoutNoFrameskip-v4",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_rainbow(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_rainbow(gpu=0)


class A3CFF(chainer.ChainList, agents.a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = links.NIPSDQNHead()
        self.pi = policies.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_functions.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["final"],
    }
))
class TestLoadA3C(unittest.TestCase):

    def _test_load_a3c(self, gpu):
        model = A3CFF(4)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4,
                                         eps=1e-1,
                                         alpha=0.99)
        opt.setup(model)
        agent = agents.A3C(model, opt, t_max=5, gamma=0.99,
                           beta=1e-2, phi=lambda x: x)
        model, exists = download_model("A3C", "BreakoutNoFrameskip-v4",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_a3c(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_a3c(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadDDPG(unittest.TestCase):
    explorer = explorers.AdditiveGaussian(scale=0.1,
                                          low=[-1., -1., -1.],
                                          high=[1., 1., 1.])

    def _test_load_ddpg(self, gpu):

        def concat_obs_and_action(obs, action):
            return F.concat((obs, action), axis=-1)

        action_size = 3
        winit = chainer.initializers.LeCunUniform(3 ** -0.5)
        q_func = chainer.Sequential(
            concat_obs_and_action,
            L.Linear(None, 400, initialW=winit),
            F.relu,
            L.Linear(None, 300, initialW=winit),
            F.relu,
            L.Linear(None, 1, initialW=winit),)
        policy = chainer.Sequential(
            L.Linear(None, 400, initialW=winit),
            F.relu,
            L.Linear(None, 300, initialW=winit),
            F.relu,
            L.Linear(None, action_size, initialW=winit),
            F.tanh,
            chainerrl.distribution.ContinuousDeterministicDistribution,)
        from chainerrl.agents.ddpg import DDPGModel
        model = DDPGModel(q_func=q_func, policy=policy)

        obs_low = [-np.inf] * 11
        fake_obs = chainer.Variable(
            model.xp.zeros_like(
                obs_low, dtype=np.float32)[None], name='observation')
        fake_action = chainer.Variable(
            model.xp.zeros_like([-1., -1., -1.], dtype=np.float32)[None],
            name='action')
        policy(fake_obs)
        q_func(fake_obs, fake_action)

        opt_a = optimizers.Adam()
        opt_c = optimizers.Adam()
        opt_a.setup(model['policy'])
        opt_c.setup(model['q_function'])

        explorer = explorers.AdditiveGaussian(scale=0.1,
                                              low=[-1., -1., -1.],
                                              high=[1., 1., 1.])

        agent = agents.DDPG(
            model,
            opt_a,
            opt_c,
            replay_buffer.ReplayBuffer(100),
            gamma=0.99,
            explorer=explorer,
            replay_start_size=1000,
            target_update_method='soft',
            target_update_interval=1,
            update_interval=1,
            soft_update_tau=5e-3,
            n_times_update=1,
            gpu=gpu,
            minibatch_size=100,
            burnin_action_func=None)

        model, exists = download_model("DDPG", "Hopper-v2",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_ddpg(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_ddpg(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadTRPO(unittest.TestCase):

    def test_load_trpo(self):
        winit = chainerrl.initializers.Orthogonal(1.)
        winit_last = chainerrl.initializers.Orthogonal(1e-2)
        action_size = 3
        policy = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, action_size, initialW=winit_last),
            policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 1, initialW=winit),
        )
        vf_opt = chainer.optimizers.Adam()
        vf_opt.setup(vf)

        agent = agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            update_interval=5000,
            max_kl=0.01,
            conjugate_gradient_max_iter=20,
            conjugate_gradient_damping=1e-1,
            gamma=0.995,
            lambd=0.97,
            vf_epochs=5,
            entropy_coef=0)

        model, exists = download_model("TRPO", "Hopper-v2",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["final"],
    }
))
class TestLoadPPO(unittest.TestCase):

    def _test_load_ppo(self, gpu):
        winit = chainerrl.initializers.Orthogonal(1.)
        winit_last = chainerrl.initializers.Orthogonal(1e-2)
        action_size = 3
        policy = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, action_size, initialW=winit_last),
            policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 1, initialW=winit))

        model = links.Branched(policy, vf)

        opt = chainer.optimizers.Adam(3e-4, eps=1e-5)
        opt.setup(model)

        agent = agents.PPO(
            model,
            opt,
            obs_normalizer=None,
            gpu=gpu,
            update_interval=2048,
            minibatch_size=64,
            epochs=10,
            clip_eps_vf=None,
            entropy_coef=0,
            standardize_advantages=True,
            gamma=0.995,
            lambd=0.97)

        model, exists = download_model("PPO", "Hopper-v2",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_ppo(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_ppo(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadTD3(unittest.TestCase):

    def _test_load_td3(self, gpu):
        def concat_obs_and_action(obs, action):
            """Concat observation and action to feed the critic."""
            return F.concat((obs, action), axis=-1)

        def make_q_func_with_optimizer():
            q_func = chainer.Sequential(
                concat_obs_and_action,
                L.Linear(None, 400, initialW=winit),
                F.relu,
                L.Linear(None, 300, initialW=winit),
                F.relu,
                L.Linear(None, 1, initialW=winit),
            )
            q_func_optimizer = optimizers.Adam().setup(q_func)
            return q_func, q_func_optimizer

        winit = chainer.initializers.LeCunUniform(3 ** -0.5)

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        action_size = 3
        policy = chainer.Sequential(
            L.Linear(None, 400, initialW=winit),
            F.relu,
            L.Linear(None, 300, initialW=winit),
            F.relu,
            L.Linear(None, action_size, initialW=winit),
            F.tanh,
            chainerrl.distribution.ContinuousDeterministicDistribution,)

        policy_optimizer = optimizers.Adam().setup(policy)

        rbuf = replay_buffer.ReplayBuffer(100)
        explorer = explorers.AdditiveGaussian(scale=0.1,
                                              low=[-1., -1., -1.],
                                              high=[1., 1., 1.])

        agent = agents.TD3(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=0.99,
            soft_update_tau=5e-3,
            explorer=explorer,
            replay_start_size=10000,
            gpu=gpu,
            minibatch_size=100,
            burnin_action_func=None)

        model, exists = download_model("TD3", "Hopper-v2",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_td3(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_td3(gpu=0)


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadSAC(unittest.TestCase):

    def _test_load_sac(self, gpu):

        winit = chainer.initializers.GlorotUniform()
        winit_policy_output = chainer.initializers.GlorotUniform(1.0)

        def concat_obs_and_action(obs, action):
            """Concat observation and action to feed the critic."""
            return F.concat((obs, action), axis=-1)

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == 3 * 2
            mean, log_scale = F.split_axis(x, 2, axis=1)
            log_scale = F.clip(log_scale, -20., 2.)
            var = F.exp(log_scale * 2)
            return chainerrl.distribution.SquashedGaussianDistribution(
                mean, var=var)

        policy = chainer.Sequential(
            L.Linear(None, 256, initialW=winit),
            F.relu,
            L.Linear(None, 256, initialW=winit),
            F.relu,
            L.Linear(None, 3 * 2, initialW=winit_policy_output),
            squashed_diagonal_gaussian_head,
        )
        policy_optimizer = optimizers.Adam(3e-4).setup(policy)

        def make_q_func_with_optimizer():
            q_func = chainer.Sequential(
                concat_obs_and_action,
                L.Linear(None, 256, initialW=winit),
                F.relu,
                L.Linear(None, 256, initialW=winit),
                F.relu,
                L.Linear(None, 1, initialW=winit),
            )
            q_func_optimizer = optimizers.Adam(3e-4).setup(q_func)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        agent = agents.SoftActorCritic(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffer.ReplayBuffer(100),
            gamma=0.99,
            replay_start_size=1000,
            gpu=gpu,
            minibatch_size=256,
            burnin_action_func=None,
            entropy_target=-3,
            temperature_optimizer=optimizers.Adam(3e-4),
        )

        model, exists = download_model("SAC", "Hopper-v2",
                                       model_type=self.pretrained_type)
        agent.load(model)
        if os.environ.get('CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED'):
            assert exists

    def test_cpu(self):
        self._test_load_sac(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_sac(gpu=0)
