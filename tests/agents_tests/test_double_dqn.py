from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from chainerrl.agents.double_dqn import DoubleDQN
from test_dqn_like import _TestDQNOnContinuousABC
from test_dqn_like import _TestDQNOnDiscreteABC
from test_dqn_like import _TestDQNOnDiscretePOABC


class TestDoubleDQNOnDiscreteABC(_TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100)


class TestDoubleDQNOnContinuousABC(_TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100)


class TestDoubleDQNOnDiscretePOABC(_TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            episodic_update=True)
