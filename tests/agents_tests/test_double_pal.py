from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()

from chainerrl.agents.double_pal import DoublePAL
from test_dqn_like import _TestDQNOnContinuousABC
from test_dqn_like import _TestDQNOnDiscreteABC
from test_dqn_like import _TestDQNOnDiscretePOABC


class TestDoublePALOnDiscreteABC(_TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100)


class TestDoublePALOnContinuousABC(_TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100)


class TestDoublePALOnDiscretePOABC(_TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            episodic_update=True)
