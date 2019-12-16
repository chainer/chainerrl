import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin
import chainerrl
from chainerrl.agents import double_iqn
from chainerrl.agents import iqn


@testing.parameterize(*testing.product({
    'quantile_thresholds_N': [1, 5],
    'quantile_thresholds_N_prime': [1, 7],
}))
class TestDoubleIQNOnDiscreteABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.ImplicitQuantileQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                iqn.CosineBasisLinear(32, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return double_iqn.DoubleIQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            quantile_thresholds_N=self.quantile_thresholds_N,
            quantile_thresholds_N_prime=self.quantile_thresholds_N_prime,
            act_deterministically=True,
        )


class TestDoubleIQNOnDiscretePOABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.StatelessRecurrentImplicitQuantileQFunction(
            psi=chainerrl.links.StatelessRecurrentSequential(
                L.Linear(obs_size, hidden_size),
                F.relu,
                L.NStepRNNTanh(1, hidden_size, hidden_size, 0),
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(32, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, env.action_space.n,
                       initialW=chainer.initializers.LeCunNormal(1e-1)),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return double_iqn.DoubleIQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            quantile_thresholds_N=32,
            quantile_thresholds_N_prime=32,
            recurrent=True,
            act_deterministically=True,
        )
