import functools
import unittest

import chainer
import chainer.functions as F
from chainer import links as L
from chainer import optimizers
from chainer import testing
import numpy as np

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
class TestLoadTRPO(unittest.TestCase):

    def _test_load_trpo(self, gpu):
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

        agent = chainerrl.agents.TRPO(
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

        model, exists = download_model("TRPO", "BreakoutNoFrameskip-v4",
                                       model_type=self.pretrained_type)
        agent.load(model)
        assert exists

    def test_cpu(self):
        self._test_load_trpo(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_trpo(gpu=0)
