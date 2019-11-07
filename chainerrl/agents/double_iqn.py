import chainer
import chainer.functions as F

from chainerrl.agents import iqn


class DoubleIQN(iqn.IQN):

    """IQN with DoubleDQN-like target computation.

    For computing targets, rather than have the target network
    output the Q-value of its highest-valued action, the
    target network outputs the Q-value of the primary network's
    highest valued action.
    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions.

        Returns:
            chainer.Variable: (batch_size, N_prime).
        """
        batch_next_state = exp_batch['next_state']
        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        batch_size = len(exp_batch['reward'])
        taus_tilde = self.xp.random.uniform(
            0, 1, size=(batch_size, self.quantile_thresholds_K)).astype('f')
        with chainer.using_config('train', False):
            if self.recurrent:
                next_tau2av, _ = self.model.n_step_forward(
                    batch_next_state,
                    exp_batch['next_recurrent_state'],
                    output_mode='concat',
                )
            else:
                next_tau2av = self.model(batch_next_state)
        greedy_actions = next_tau2av(taus_tilde).greedy_actions
        taus_prime = self.xp.random.uniform(
            0, 1,
            size=(batch_size, self.quantile_thresholds_N_prime)).astype('f')
        if self.recurrent:
            target_next_tau2av, _ = self.target_model.n_step_forward(
                batch_next_state,
                exp_batch['next_recurrent_state'],
                output_mode='concat',
            )
        else:
            target_next_tau2av = self.target_model(batch_next_state)
        target_next_maxz = target_next_tau2av(
            taus_prime).evaluate_actions_as_quantiles(greedy_actions)

        batch_discount = exp_batch['discount']
        assert batch_rewards.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)
        assert batch_discount.shape == (batch_size,)
        batch_rewards = F.broadcast_to(
            batch_rewards[..., None], target_next_maxz.shape)
        batch_terminal = F.broadcast_to(
            batch_terminal[..., None], target_next_maxz.shape)
        batch_discount = F.broadcast_to(
            batch_discount[..., None], target_next_maxz.shape)

        return (batch_rewards
                + batch_discount * (1.0 - batch_terminal) * target_next_maxz)
