from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import chainer.functions as F

from chainerrl.agents.dqn import DQN
from chainerrl.functions import scale_grad


class ResidualDQN(DQN):
    """DQN that allows maxQ also backpropagate gradients."""

    def __init__(self, *args, **kwargs):
        self.grad_scale = kwargs.pop('grad_scale', 1.0)
        super().__init__(*args, **kwargs)

    def sync_target_network(self):
        pass

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        target_next_qout = self.q_function(batch_next_state, test=False)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):

        batch_state = exp_batch['state']
        batch_size = len(batch_state)

        # Compute Q-values for current states
        qout = self.q_function(batch_state, test=False)

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        # Target values must also backprop gradients
        batch_q_target = F.reshape(
            self._compute_target_values(exp_batch, gamma), (batch_size, 1))

        return batch_q, scale_grad.scale_grad(batch_q_target, self.grad_scale)

    @property
    def saved_attributes(self):
        # ResidualDQN doesn't use target models
        return ('model', 'optimizer')

    def input_initial_batch_to_target_model(self, batch):
        pass
