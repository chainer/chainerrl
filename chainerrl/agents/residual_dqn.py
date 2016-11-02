from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda

from agents.dqn import DQN
from functions import scale_grad


class ResidualDQN(DQN):
    """DQN that allows maxQ also backpropagate gradients.
    """

    def __init__(self, *args, **kwargs):
        self.grad_scale = kwargs.pop('grad_scale', 1.0)
        super().__init__(*args, **kwargs)

    def sync_target_network(self):
        pass

    def _compute_target_values(self, experiences, gamma):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        target_next_qout = self.q_function(batch_next_state, test=False)
        next_q_max = target_next_qout.max

        batch_rewards = self.xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32)

        batch_terminal = self.xp.asarray(
            [elem['is_state_terminal'] for elem in experiences],
            dtype=np.float32)

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, experiences, gamma):

        batch_size = len(experiences)

        # Compute Q-values for current states
        batch_state = self._batch_states(
            [elem['state'] for elem in experiences])

        qout = self.q_function(batch_state, test=False)
        xp = cuda.get_array_module(qout.greedy_actions.data)

        batch_actions = chainer.Variable(
            xp.asarray([elem['action'] for elem in experiences]))
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        batch_q_target = F.reshape(
            self._compute_target_values(experiences, gamma), (batch_size, 1))

        return batch_q, scale_grad.scale_grad(batch_q_target, self.grad_scale)
