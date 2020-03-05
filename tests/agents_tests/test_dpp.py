from chainer import testing

import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin
from chainerrl.agents.dpp import DPP
from chainerrl.agents.dpp import DPPGreedy
from chainerrl.agents.dpp import DPPL


def parse_dpp_agent(dpp_type):
    return {'DPP': DPP,
            'DPPL': DPPL,
            'DPPGreedy': DPPGreedy}[dpp_type]


@testing.parameterize(
    *testing.product({
        'dpp_type': ['DPP', 'DPPL', 'DPPGreedy'],
    })
)
class TestDPPOnDiscreteABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


# DPP and DPPL don't support continuous action spaces
@testing.parameterize(
    *testing.product({
        'dpp_type': ['DPPGreedy'],
    })
)
class TestDPPOnContinuousABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100)


@testing.parameterize(
    *testing.product({
        'dpp_type': ['DPP', 'DPPL', 'DPPGreedy'],
    })
)
class TestDPPOnDiscretePOABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            recurrent=True)
