import chainer
from chainer import functions as F

from chainerrl.agents import dqn


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

    def _compute_y_and_t(self, exp_batch):

        batch_state = exp_batch['state']
        batch_size = len(exp_batch['reward'])

        if self.recurrent:
            qout, _ = self.model.n_step_forward(
                batch_state, exp_batch['recurrent_state'],
                output_mode='concat')
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch['action']

        batch_q = qout.evaluate_actions(batch_actions)

        # Compute target values
        batch_next_state = exp_batch['next_state']

        with chainer.no_backprop_mode():
            if self.recurrent:
                target_qout, _ = self.target_model.n_step_forward(
                    batch_state, exp_batch['recurrent_state'],
                    output_mode='concat')
                target_next_qout, _ = self.target_model.n_step_forward(
                    batch_next_state, exp_batch['next_recurrent_state'],
                    output_mode='concat')
            else:
                target_qout = self.target_model(batch_state)
                target_next_qout = self.target_model(batch_next_state)

            next_q_max = F.reshape(target_next_qout.max, (batch_size,))

            batch_rewards = exp_batch['reward']
            batch_terminal = exp_batch['is_state_terminal']

            # T Q: Bellman operator
            t_q = batch_rewards + exp_batch['discount'] * \
                (1.0 - batch_terminal) * next_q_max

            # T_AL Q: advantage learning operator
            cur_advantage = F.reshape(
                target_qout.compute_advantage(batch_actions), (batch_size,))
            tal_q = t_q + self.alpha * cur_advantage

        return batch_q, tal_q
