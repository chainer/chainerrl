from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import chainer

from chainerrl.agents import dqn
from chainerrl.recurrent import state_kept


class DoubleDQN(dqn.DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        with chainer.using_config('train', False):
            with state_kept(self.q_function):
                next_qout = self.q_function(batch_next_state)

        target_next_qout = self.target_q_function(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max
