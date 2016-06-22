from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from agents.double_pal import DoublePAL
import replay_buffer
from test_dqn_like import _TestDQNLike


class TestDoublePAL(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DoublePAL(q_func, opt, rbuf, gpu=gpu, gamma=0.9,
                         explorer=explorer, replay_start_size=100,
                         target_update_frequency=100)
