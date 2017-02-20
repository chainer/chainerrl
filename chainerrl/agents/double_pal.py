from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F

from chainerrl.agents import pal
from chainerrl.recurrent import state_kept


class DoublePAL(pal.PAL):

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_state = exp_batch['state']
        batch_size = len(exp_batch['reward'])

        qout = self.q_function(batch_state, test=False)

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions(batch_actions)

        # Compute target values

        with chainer.no_backprop_mode():
            target_qout = self.target_q_function(batch_state, test=True)

            batch_next_state = exp_batch['next_state']

            with state_kept(self.q_function):
                next_qout = self.q_function(batch_next_state, test=False)

            with state_kept(self.target_q_function):
                target_next_qout = self.target_q_function(
                    batch_next_state, test=True)
            next_q_max = F.reshape(target_next_qout.evaluate_actions(
                next_qout.greedy_actions), (batch_size,))

            batch_rewards = exp_batch['reward']
            batch_terminal = exp_batch['is_state_terminal']

            # T Q: Bellman operator
            t_q = batch_rewards + self.gamma * \
                (1.0 - batch_terminal) * next_q_max

            # T_PAL Q: persistent advantage learning operator
            cur_advantage = F.reshape(
                target_qout.compute_advantage(batch_actions), (batch_size,))
            next_advantage = F.reshape(
                target_next_qout.compute_advantage(batch_actions),
                (batch_size,))
            tpal_q = t_q + self.alpha * \
                F.maximum(cur_advantage, next_advantage)

        return batch_q, tpal_q
