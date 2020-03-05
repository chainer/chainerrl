import chainer
from chainer import functions as F

from chainerrl.agents import pal


class DoublePAL(pal.PAL):

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

        with chainer.no_backprop_mode():
            batch_next_state = exp_batch['next_state']
            if self.recurrent:
                next_qout, _ = self.model.n_step_forward(
                    batch_next_state, exp_batch['next_recurrent_state'],
                    output_mode='concat')
                target_qout, _ = self.target_model.n_step_forward(
                    batch_state, exp_batch['recurrent_state'],
                    output_mode='concat')
                target_next_qout, _ = self.target_model.n_step_forward(
                    batch_next_state, exp_batch['next_recurrent_state'],
                    output_mode='concat')
            else:
                next_qout = self.model(batch_next_state)
                target_qout = self.target_model(batch_state)
                target_next_qout = self.target_model(batch_next_state)

            next_q_max = F.reshape(target_next_qout.evaluate_actions(
                next_qout.greedy_actions), (batch_size,))

            batch_rewards = exp_batch['reward']
            batch_terminal = exp_batch['is_state_terminal']

            # T Q: Bellman operator
            t_q = batch_rewards + exp_batch['discount'] * \
                (1.0 - batch_terminal) * next_q_max

            # T_PAL Q: persistent advantage learning operator
            cur_advantage = F.reshape(
                target_qout.compute_advantage(batch_actions), (batch_size,))
            next_advantage = F.reshape(
                target_next_qout.compute_advantage(batch_actions),
                (batch_size,))
            tpal_q = t_q + self.alpha * \
                F.maximum(cur_advantage, next_advantage)

        return batch_q, tpal_q
