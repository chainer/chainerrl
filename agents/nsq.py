from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
from logging import getLogger
logger = getLogger(__name__)
import copy
import os

import numpy as np
import chainer
from chainer import functions as F
from chainer import serializers

import copy_param
import async


class NSQ(object):
    """N-step Q-Learning.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, process_idx, q_function, target_q_function, optimizer,
                 t_max, gamma, i_target, explorer,
                 clip_reward=True):

        self.process_idx = process_idx

        self.shared_q_function = q_function
        self.target_q_function = target_q_function
        self.q_function = copy.deepcopy(self.shared_q_function)

        async.assert_params_not_shared(self.shared_q_function, self.q_function)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.explorer = explorer
        self.i_target = i_target
        self.clip_reward = clip_reward
        self.t = 0
        self.t_start = 0
        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.q_function,
                              source_link=self.shared_q_function)

    def act(self, t, state, reward, is_state_terminal):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        state = state.reshape((1,) + state.shape)

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            # Update
            if is_state_terminal:
                R = 0
            else:
                R = float(self.target_q_function(state).max.data)

            loss = 0
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                q = F.reshape(self.past_action_values[i], (1, 1))
                # Accumulate gradients of Q-function
                # loss += (R - q) ** 2
                # loss += F.mean_squared_error(q, chainer.Variable(np.asarray([R])))
                loss += F.sum(F.huber_loss(
                    q, chainer.Variable(np.asarray([[R]], dtype=np.float32)),
                    delta=1.0))

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
            qout = self.q_function(state)
            action = self.explorer.select_action(
                self.t, lambda: qout.greedy_actions.data[0])
            q = qout.evaluate_actions(np.asarray([action]))
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

            return action
        else:
            return None

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.shared_q_function)
        # TODO(fujita): Save/load a target model separately for resuming
        serializers.load_hdf5(model_filename, self.target_q_function)
        self.sync_parameters()
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.q_function)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
