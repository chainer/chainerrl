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
import multiprocessing as mp

import numpy as np
import chainer
from chainer import functions as F
from chainer import serializers

import copy_param
import async
from chainerrl.misc.makedirs import makedirs


class NSQ(object):
    """N-step Q-Learning.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, process_idx, q_function, optimizer,
                 t_max, gamma, i_target, explorer, phi=lambda x: x,
                 clip_reward=True):

        self.process_idx = process_idx

        self.shared_q_function = q_function
        self.target_q_function = copy.deepcopy(q_function)
        self.q_function = copy.deepcopy(self.shared_q_function)

        async.assert_params_not_shared(self.shared_q_function, self.q_function)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.explorer = explorer
        self.i_target = i_target
        self.clip_reward = clip_reward
        self.phi = phi
        self.t_global = mp.Value('l', 0)

        self.t = 0
        self.t_start = 0
        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.q_function,
                              source_link=self.shared_q_function)

    @property
    def shared_attributes(self):
        return ('shared_q_function', 'target_q_function', 'optimizer',
                't_global')

    def update(self, statevar):
        assert self.t_start < self.t

        # Update
        if statevar is None:
            R = 0
        else:
            R = float(self.target_q_function(statevar).max.data)

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

    def act_and_train(self, obs, reward):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        statevar = chainer.Variable(np.expand_dims(self.phi(obs), 0))

        self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        qout = self.q_function(statevar)
        action = self.explorer.select_action(
            self.t, lambda: qout.greedy_actions.data[0])
        q = qout.evaluate_actions(np.asarray([action]))
        if self.t % 100 == 0:
            logger.debug('q:%s', q.data)
        self.past_action_values[self.t] = q
        self.t += 1
        with self.t_global.get_lock():
            self.t_global.value += 1
            t_global = self.t_global.value

        if t_global % self.i_target == 0:
            logger.debug('target synchronized t_global:%s t_local:%s',
                         t_global, self.t)
            copy_param.copy_param(self.target_q_function, self.q_function)

        return action

    def act(self, obs):
        statevar = chainer.Variable(np.expand_dims(self.phi(obs), 0))
        qout = self.q_function(statevar)
        return qout.greedy_actions.data[0]

    def stop_episode_and_train(self, state, reward, done=False):
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        self.past_rewards[self.t - 1] = reward
        if done:
            self.update(None)
        else:
            statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))
            self.update(statevar)

    def save(self, dirname):
        makedirs(dirname, exist_ok=True)
        serializers.save_npz(os.path.join(dirname, 'q_function.npz'),
                             self.q_function)
        serializers.save_npz(os.path.join(dirname, 'target_q_function.npz'),
                             self.target_q_function)
        serializers.save_npz(os.path.join(dirname, 'optimizer.npz'),
                             self.optimizer)

    def load(self, dirname):
        serializers.load_npz(os.path.join(
            dirname, 'q_function.npz'), self.q_function)

        target_filename = os.path.join(dirname, 'target_q_function.npz')
        if os.path.exists(target_filename):
            serializers.load_npz(target_filename, self.target_q_function)
        else:
            print('WARNING: {0} was not found'.format(target_filename))
            copy_param.copy_param(target_link=self.target_q_function,
                                  source_link=self.q_function)

        opt_filename = os.path.join(dirname, 'optimizer.npz')
        if os.path.exists(opt_filename):
            serializers.load_npz(opt_filename, self.optimizer)
        else:
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
