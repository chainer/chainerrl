import chainer

from chainerrl.agents import categorical_dqn
from chainerrl.agents.categorical_dqn import _apply_categorical_projection


class CategoricalDoubleDQN(categorical_dqn.CategoricalDQN):
    """Categorical Double DQN.

    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch['next_state']
        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        with chainer.using_config('train', False):
            if self.recurrent:
                target_next_qout, _ = self.target_model.n_step_forward(
                    batch_next_state, exp_batch['next_recurrent_state'],
                    output_mode='concat')
                next_qout, _ = self.model.n_step_forward(
                    batch_next_state, exp_batch['next_recurrent_state'],
                    output_mode='concat')
            else:
                target_next_qout = self.target_model(batch_next_state)
                next_qout = self.model(batch_next_state)

        batch_size = batch_rewards.shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.size

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.evaluate_actions_as_distribution(
            next_qout.greedy_actions.array).array
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (batch_rewards[..., None]
              + (1.0 - batch_terminal[..., None])
              * self.xp.expand_dims(exp_batch['discount'], 1)
              * z_values[None])
        return _apply_categorical_projection(Tz, next_q_max, z_values)
