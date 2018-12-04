from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import dqn


class SARSA(dqn.DQN):
    """SARSA.

    Unlike DQN, this agent uses actions that have been actually taken to
    compute target Q values, thus is an on-policy algorithm.
    """

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch['next_state']
        batch_next_action = exp_batch['next_action']

        next_target_action_value = self.target_q_function(
            batch_next_state)
        next_q = next_target_action_value.evaluate_actions(
            batch_next_action)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q

    def batch_act_and_train(self, batch_obs):
        raise NotImplementedError('SARSA does not support batch training')

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        raise NotImplementedError('SARSA does not support batch training')
