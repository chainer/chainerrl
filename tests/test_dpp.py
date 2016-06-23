from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from agents.dpp import DPP
import replay_buffer
from test_dqn_like import _TestDQNLike
from chainer import testing


@testing.parameterize(
    {'eta': 1e-2},
    {'eta': 1e-1},
    {'eta': 1e-0},
    {'eta': 1e+1},
)
class TestDQN(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DPP(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_frequency=100,
                   eta=self.eta)
