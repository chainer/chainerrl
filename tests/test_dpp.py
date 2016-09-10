from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from agents.dpp import DPP
from agents.dpp import DPPL
from agents.dpp import DPPGreedy
import replay_buffer
from test_dqn_like import _TestDQNLike


class TestDPP(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DPP(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_frequency=100)

    def test_abc_continuous_gpu(self):
        print("DPP doesn't support continuous action spaces.")

    def test_abc_continuous_cpu(self):
        print("DPP doesn't support continuous action spaces.")


class TestDPPL(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DPPL(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                    replay_start_size=100, target_update_frequency=100)

    def test_abc_continuous_gpu(self):
        print("DPPL doesn't support continuous action spaces.")

    def test_abc_continuous_cpu(self):
        print("DPPL doesn't support continuous action spaces.")


class TestDPPGreedy(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DPPGreedy(q_func, opt, rbuf, gpu=gpu, gamma=0.9,
                         explorer=explorer,
                         replay_start_size=100, target_update_frequency=100)
