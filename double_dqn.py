import dqn
from chainer import cuda


class DoubleDQN(dqn.DQN):

    def _compute_target_values(self, experiences, gamma, batch_q):

        batch_next_state = self._batch_states(
            [elem['next_state'] for elem in experiences])

        batch_next_q_by_model = cuda.to_cpu(
            self.q_function.forward(batch_next_state, test=True).data)

        batch_next_q_by_target_model = cuda.to_cpu(
            self.target_q_function.forward(batch_next_state, test=True).data)

        batch_q_target = batch_q.copy()

        # Set target values for max actions
        for batch_idx in xrange(len(experiences)):
            experience = experiences[batch_idx]
            action = experience['action']
            reward = experience['reward']
            max_action = batch_next_q_by_model[batch_idx].argmax()
            max_q = batch_next_q_by_target_model[batch_idx, max_action]
            if experience['is_state_terminal']:
                q_target = reward
            else:
                q_target = reward + self.gamma * max_q
            q_target = reward + gamma * max_q
            batch_q_target[batch_idx, action] = q_target

        return batch_q_target
