from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from chainerrl.agents.residual_dqn import ResidualDQN
from test_dqn_like import _TestDQNOnContinuousABC
from test_dqn_like import _TestDQNOnDiscreteABC
from test_dqn_like import _TestDQNOnDiscretePOABC


class TestResidualDQNOnDiscreteABC(_TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            grad_scale=1e-1)


class TestResidualDQNOnContinuousABC(_TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            grad_scale=1e-1)


class TestResidualDQNOnDiscretePOABC(_TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            episodic_update=True,
            grad_scale=1e-1)
