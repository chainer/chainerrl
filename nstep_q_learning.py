import copy
from logging import getLogger
logger = getLogger(__name__)

import numpy as np
import chainer
from chainer import functions as F

import agent
import copy_param
import smooth_l1_loss


class NStepQLearning(agent.Agent):
    """N-step Q-Learning.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, q_function, optimizer, t_max, gamma, epsilon, i_target):
        # Globally shared model
        self.shared_q_function = q_function
        # Thread specific model
        self.q_function = copy.deepcopy(q_function)
        self.target_q_function = copy.deepcopy(q_function)
        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.epsilon = epsilon
        self.i_target = i_target
        self.t = 0
        self.t_start = 0
        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}

    def sync_parameters(self):
        copy_param.copy_param(self.q_function, self.shared_q_function)

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
                R = float(
                    self.target_q_function.sample_greedily_with_value(state)[1].data)

            loss = 0
            for i in reversed(xrange(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                q = F.reshape(self.past_action_values[i], (1, 1))
                # Accumulate gradients of Q-function
                # loss += (R - q) ** 2
                # loss += F.mean_squared_error(q, chainer.Variable(np.asarray([R])))
                loss += smooth_l1_loss.smooth_l1_loss(
                    q, chainer.Variable(np.asarray([[R]], dtype=np.float32)))

            # Do we need to normalize losses by (self.t - self.t_start)?
            # Otherwise, loss scales can be different in case of self.t_max
            # and in case of termination.

            # I'm not sure but if we need to normalize losses...
            # loss /= self.t - self.t_start

            # Compute gradients using thread-specific model
            self.q_function.zerograds()
            loss.backward()
            # Copy the gradients to the globally shared model
            self.shared_q_function.zerograds()
            copy_param.copy_grad(self.shared_q_function, self.q_function)
            # Update the globally shared model
            self.optimizer.update()

            self.sync_parameters()

            self.past_action_values = {}
            self.past_states = {}
            self.past_rewards = {}

            self.t_start = self.t

        if not is_state_terminal:
            self.past_states[self.t] = state
            action, q = self.q_function.sample_epsilon_greedily_with_value(
                state, self.epsilon)
            if self.t % 100 == 0:
                logger.debug('q:%s', q.data)
            self.past_action_values[self.t] = q
            self.t += 1

            # Update the target network
            # Global counter T is used in the original paper, but here we use
            # process specific counter instead. So i_target should be set
            # x-times smaller, where x is the number of processes
            if self.t % self.i_target == 0:
                logger.debug('self.t:%s', self.t)
                copy_param.copy_param(self.target_q_function, self.q_function)

            return action[0]
        else:
            return None

    @property
    def links(self):
        return [self.shared_q_function, self.target_q_function]

    @property
    def optimizers(self):
        return [self.optimizer]
