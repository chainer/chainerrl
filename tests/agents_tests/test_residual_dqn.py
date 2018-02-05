from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import basetest_dqn_like as base
from chainerrl.agents.residual_dqn import ResidualDQN


class TestResidualDQNOnDiscreteABC(base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            grad_scale=1e-1)


class TestResidualDQNOnContinuousABC(base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            grad_scale=1e-1)


class TestResidualDQNOnDiscretePOABC(base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            episodic_update=True,
            grad_scale=1e-1)
