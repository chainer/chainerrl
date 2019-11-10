from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin
import chainerrl
from chainerrl.agents import fqf


@testing.parameterize(*testing.product({
    'N': [2, 32],
}))
class TestFQFOnDiscreteABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return fqf.FQQFunction(
            psi=chainerrl.links.Sequence(
                L.Linear(obs_size, hidden_size),
                F.relu,
            ),
            phi=chainerrl.links.Sequence(
                chainerrl.agents.iqn.CosineBasisLinear(32, hidden_size),
                F.relu,
            ),
            f=L.Linear(hidden_size, env.action_space.n),
            proposal_net=L.Linear(
                hidden_size,
                self.N,
                initialW=chainer.initializers.LeCunNormal(1e-2),
            ),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return fqf.FQF(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
        )


class TestFQFOnDiscretePOABC(
        _TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC):

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return fqf.StatelessRecurrentFQQFunction(
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
            proposal_net=L.Linear(
                hidden_size,
                17,
                initialW=chainer.initializers.LeCunNormal(1e-2),
            ),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return fqf.FQF(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_interval=100,
            recurrent=True,
        )
