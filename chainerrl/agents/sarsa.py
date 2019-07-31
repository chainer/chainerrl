from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import dqn


class SARSA(dqn.DQN):
    """Off-policy SARSA.

    This agent learns the Q-function of a behavior policy defined via the
    given explorer, not the optimal policy.
    """

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch['next_state']

        if self.recurrent:
            target_next_qout, _ = self.target_model.n_step_forward(
                batch_next_state, exp_batch['next_recurrent_state'],
                output_mode='concat')
        else:
            target_next_qout = self.target_model(batch_next_state)
        # Choose an action using the behavior policy
        batch_next_action = self.xp.array([
            self.explorer.select_action(
                self.t, lambda: target_next_qout.greedy_actions.array[i],
                action_value=target_next_qout[i:i + 1],
            )
            for i in range(len(exp_batch['action']))])
        next_q = target_next_qout.evaluate_actions(batch_next_action)
        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q
