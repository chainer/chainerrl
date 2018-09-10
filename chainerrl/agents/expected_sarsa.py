from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import dqn
import chainer.functions as F

class ExpectedSARSA(dqn.DQN):
    """SARSA.

    Unlike DQN, this agent uses actions that have been actually taken to
    compute tareget Q values, thus is an on-policy algorithm.
    """

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']
        batch_next_action = exp_batch['next_action']

        next_target_action_value = self.target_q_function(
            batch_next_state)
        next_q = next_target_action_value.evaluate_actions(
            batch_next_action)
        greedy = next_target_action_value.greedy_actions
        values = next_target_action_value.q_values

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        max_prob = 1-self.explorer.epsilon
        pi_dist = self.xp.ones_like(values) * (self.explorer.epsilon/values.shape[1])
        pi_dist[self.xp.arange(pi_dist.shape[0]), greedy.data] += max_prob

        if self.head:
            mean = batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q
            sigma = self.gamma * next_target_action_value.max_sigma
            return mean, sigma[:, None]
        else:
            expected_q = F.sum(pi_dist * values, 1)
            return batch_rewards + self.gamma * (1.0 - batch_terminal) * expected_q
