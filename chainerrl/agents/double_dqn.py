import chainer

from chainerrl.agents import dqn


class DoubleDQN(dqn.DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch['next_state']

        with chainer.using_config('train', False):
            if self.recurrent:
                next_qout, _ = self.model.n_step_forward(
                    batch_next_state,
                    exp_batch['next_recurrent_state'],
                    output_mode='concat',
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = self.target_model.n_step_forward(
                batch_next_state,
                exp_batch['next_recurrent_state'],
                output_mode='concat',
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max
