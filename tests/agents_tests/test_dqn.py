from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()

import basetest_dqn_like as base
import chainerrl
from chainerrl.agents.dqn import DQN


class TestDQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnDiscreteABCBoltzmann(base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = chainerrl.explorers.Boltzmann()
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnContinuousABC(base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100)


class TestDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_interval=100,
                   episodic_update=True)
