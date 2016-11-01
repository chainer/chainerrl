from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from agents.residual_dqn import ResidualDQN
import replay_buffer
from test_dqn_like import _TestDQNLike
from chainer import testing


@testing.parameterize(
    {'grad_scale': 1e-1},
    {'grad_scale': 1e-2},
    {'grad_scale': 1e-3},
)
class TestDQN(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return ResidualDQN(
            q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
            replay_start_size=100, target_update_frequency=100,
            grad_scale=self.grad_scale)
