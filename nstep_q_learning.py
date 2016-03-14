from chainer import functions as F

import agent


class NStepQLearning(agent.Agent):
    """N-step Q-Learning.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, q_function, optimizer, t_max, gamma, epsilon):
        self.q_function = q_function
        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.epsilon = epsilon
        self.t = 0
        self.t_start = 0
        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}

    def act(self, state, reward, is_state_terminal):

        state = state.reshape((1,) + state.shape)

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            # Update
            if is_state_terminal:
                R = 0
            else:
                R = float(self.q_function.sample_greedily_with_value(state)[1].data)

            self.optimizer.zero_grads()

            loss = 0
            for i in reversed(xrange(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                q = self.past_action_values[i]
                # Accumulate gradients of Q-function
                loss += (R - q) ** 2

            # Do we need to normalize losses by (self.t - self.t_start)?
            # Otherwise, loss scales can be different in case of self.t_max
            # and in case of termination.

            # I'm not sure but if we need to normalize losses...
            # loss /= self.t - self.t_start

            loss.backward()

            self.optimizer.update()

            self.past_action_values = {}
            self.past_states = {}
            self.past_rewards = {}

            self.t_start = self.t

        if not is_state_terminal:
            self.past_states[self.t] = state
            action, q = self.q_function.sample_epsilon_greedily_with_value(
                state, self.epsilon)
            print 'q:{}'.format(q.data)
            self.past_action_values[self.t] = q
            self.t += 1
            return action[0]
        else:
            return None

    @property
    def links(self):
        return [self.q_function]

    @property
    def optimizers(self):
        return [self.optimizer]
