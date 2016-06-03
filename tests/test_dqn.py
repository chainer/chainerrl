from dqn import DQN
import replay_buffer
from test_dqn_like import _TestDQNLike


class TestDoubleDQN(_TestDQNLike):

    def make_agent(self, gpu, q_func, opt):
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        return DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.9, epsilon=0.1,
                   replay_start_size=1000, target_update_frequency=1000)
