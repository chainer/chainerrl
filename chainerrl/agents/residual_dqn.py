from chainerrl.agents.dqn import DQN
from chainerrl.functions import scale_grad


class ResidualDQN(DQN):
    """DQN that allows maxQ also backpropagate gradients."""

    def __init__(self, *args, **kwargs):
        self.grad_scale = kwargs.pop('grad_scale', 1.0)
        super().__init__(*args, **kwargs)

    def sync_target_network(self):
        pass

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch['next_state']
        if self.recurrent:
            target_next_qout, _ = self.model.n_step_forward(
                batch_next_state, exp_batch['next_recurrent_state'],
                output_mode='concat')
        else:
            target_next_qout = self.model(batch_next_state)

        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        batch_discount = exp_batch['discount']

        return (batch_rewards
                + batch_discount * (1.0 - batch_terminal) * next_q_max)

    def _compute_y_and_t(self, exp_batch):

        batch_state = exp_batch['state']

        # Compute Q-values for current states
        if self.recurrent:
            qout, _ = self.model.n_step_forward(
                batch_state, exp_batch['recurrent_state'],
                output_mode='concat')
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions(batch_actions)[..., None]

        # Target values must also backprop gradients
        batch_q_target = self._compute_target_values(exp_batch)[..., None]

        return batch_q, scale_grad.scale_grad(batch_q_target, self.grad_scale)

    @property
    def saved_attributes(self):
        # ResidualDQN doesn't use target models
        return ('model', 'optimizer')
