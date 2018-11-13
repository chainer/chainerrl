from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

from chainerrl.agents.double_pal import DoublePAL

import basetest_dqn_like
from basetest_training import _TestBatchTrainingMixin


class TestDoublePALOnDiscreteABC(
        _TestBatchTrainingMixin,
        basetest_dqn_like._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


class TestDoublePALOnContinuousABC(
        _TestBatchTrainingMixin,
        basetest_dqn_like._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


class TestDoublePALOnDiscretePOABC(basetest_dqn_like._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            episodic_update=True)
