from agents.dqn import DQN
import replay_buffer
from test_dqn_like import _TestDQNLike


class TestDQN(_TestDQNLike):

    def make_agent(self, gpu, q_func, explorer, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, explorer=explorer,
                   replay_start_size=100, target_update_frequency=100)
