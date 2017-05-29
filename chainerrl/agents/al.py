from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F

from chainerrl.agents import dqn
from chainerrl.recurrent import state_kept


class AL(dqn.DQN):
    """Advantage Learning.

    See: http://arxiv.org/abs/1512.04860.

    Args:
      alpha (float): Weight of (persistent) advantages. Convergence
        is guaranteed only for alpha in [0, 1).

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.9)
        super().__init__(*args, **kwargs)

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_state = exp_batch['state']
        batch_size = len(exp_batch['reward'])

        qout = self.q_function(batch_state)

        batch_actions = exp_batch['action']

        batch_q = qout.evaluate_actions(batch_actions)

        # Compute target values

        with chainer.no_backprop_mode():
            target_qout = self.target_q_function(batch_state)

            batch_next_state = exp_batch['next_state']

            with state_kept(self.target_q_function):
                target_next_qout = self.target_q_function(
                    batch_next_state)
            next_q_max = F.reshape(target_next_qout.max, (batch_size,))

            batch_rewards = exp_batch['reward']
            batch_terminal = exp_batch['is_state_terminal']

            # T Q: Bellman operator
            t_q = batch_rewards + self.gamma * \
                (1.0 - batch_terminal) * next_q_max

            # T_AL Q: advantage learning operator
            cur_advantage = F.reshape(
                target_qout.compute_advantage(batch_actions), (batch_size,))
            tal_q = t_q + self.alpha * cur_advantage

        return batch_q, tal_q

    def input_initial_batch_to_target_model(self, batch):
        pass
