from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
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
from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_functions import DistributionalDuelingDQN
from chainerrl.misc import download_model
from chainerrl import replay_buffer


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
