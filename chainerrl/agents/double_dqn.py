from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from chainerrl.agents import dqn


class DoubleDQN(dqn.DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, experiences, gamma):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        next_qout = self.q_function(batch_next_state, test=True)

        target_next_qout = self.target_q_function(batch_next_state, test=True)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_rewards = self.xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32)

        batch_terminal = self.xp.asarray(
            [elem['is_state_terminal'] for elem in experiences],
            dtype=np.float32)

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max
