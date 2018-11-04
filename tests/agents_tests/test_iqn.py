from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import chainer.functions as F
import chainer.links as L
from chainer import testing

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents import iqn


@testing.parameterize(*testing.product({
    'quantile_thresholds_N': [1, 5],
    'quantile_thresholds_N_prime': [1, 7],
}))
class TestIQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=iqn.CosineBasisLinearReLU(64, hidden_size),
            f=L.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return iqn.IQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            quantile_thresholds_N=self.quantile_thresholds_N,
            quantile_thresholds_N_prime=self.quantile_thresholds_N_prime,
        )
