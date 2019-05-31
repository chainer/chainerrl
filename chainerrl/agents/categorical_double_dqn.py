from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer

from chainerrl.agents import categorical_dqn
from chainerrl.agents.categorical_dqn import _apply_categorical_projection
from chainerrl.recurrent import state_kept


class CategoricalDoubleDQN(categorical_dqn.CategoricalDQN):
    """Categorical Double DQN.

    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch['next_state']
        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        with chainer.using_config('train', False), state_kept(self.q_function):
            next_qout = self.q_function(batch_next_state)

        target_next_qout = self.target_q_function(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_size = batch_rewards.shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.size

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.max_as_distribution.array
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (batch_rewards[..., None]
              + (1.0 - batch_terminal[..., None])
              * self.xp.expand_dims(exp_batch['discount'], 1)
              * z_values[None])
        return _apply_categorical_projection(Tz, next_q_max, z_values)
