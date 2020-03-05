import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin
from chainerrl.agents.residual_dqn import ResidualDQN


class TestResidualDQNOnDiscreteABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnDiscreteABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            grad_scale=1e-1)


class TestResidualDQNOnContinuousABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnContinuousABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            grad_scale=1e-1)


class TestResidualDQNOnDiscretePOABC(
        _TestBatchTrainingMixin,
        base._TestDQNOnDiscretePOABC):

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            recurrent=True,
            grad_scale=1e-1)
