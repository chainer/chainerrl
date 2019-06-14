from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import chainer.links as L

import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin
import chainerrl
from chainerrl.agents import CategoricalDoubleDQN


def make_distrib_ff_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
        env.observation_space.low.size, env.action_space.n,
        n_atoms=n_atoms,
        v_min=v_min,
        v_max=v_max,
        n_hidden_channels=20,
        n_hidden_layers=1,
    )


def make_distrib_recurrent_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return chainerrl.links.Sequence(
        L.LSTM(env.observation_space.low.size, 20),
        chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
            20, env.action_space.n,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            n_hidden_channels=None,
            n_hidden_layers=0,
        ),
    )


class TestCategoricalDoubleDQNOnDiscreteABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        return make_distrib_ff_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


# Continuous action spaces are not supported

class TestCategoricalDoubleDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_q_func(self, env):
        return make_distrib_recurrent_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            episodic_update=True)
