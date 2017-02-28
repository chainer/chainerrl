from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import copy
from logging import getLogger
import multiprocessing as mp

import chainer
from chainer import functions as F
import numpy as np

from chainerrl.agent import AsyncAgent
from chainerrl.agent import AttributeSavingMixin
from chainerrl.misc import async
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_kept


class NSQ(AttributeSavingMixin, AsyncAgent):
    """Asynchronous N-step Q-Learning.

    See http://arxiv.org/abs/1602.01783

    Args:
        q_function (A3CModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        i_target (intn): The target model is updated after every i_target
            global steps
        explorer (Explorer): Explorer to use in training
        phi (callable): Feature extractor function
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ['q_function', 'target_q_function', 'optimizer']

    def __init__(self, q_function, optimizer,
                 t_max, gamma, i_target, explorer, phi=lambda x: x,
                 average_q_decay=0.999, logger=getLogger(__name__),
                 batch_states=batch_states):

        self.shared_q_function = q_function
        self.target_q_function = copy.deepcopy(q_function)
        self.q_function = copy.deepcopy(self.shared_q_function)

        async.assert_params_not_shared(self.shared_q_function, self.q_function)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.explorer = explorer
        self.i_target = i_target
        self.phi = phi
        self.logger = logger
        self.average_q_decay = average_q_decay
        self.batch_states = batch_states

        self.t_global = mp.Value('l', 0)
        self.t = 0
        self.t_start = 0
        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}
        self.average_q = 0

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
            with state_kept(self.target_q_function):
                R = float(self.target_q_function(statevar).max.data)

        loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            q = F.reshape(self.past_action_values[i], (1, 1))
            # Accumulate gradients of Q-function
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
        if isinstance(self.q_function, Recurrent):
            self.q_function.unchain_backward()

        self.past_action_values = {}
        self.past_states = {}
        self.past_rewards = {}

        self.t_start = self.t

    def act_and_train(self, obs, reward):

        statevar = self.batch_states([obs], np, self.phi)

        self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        if isinstance(self.target_q_function, Recurrent):
            # Evaluate it to update states
            self.target_q_function(statevar)
        qout = self.q_function(statevar)
        action = self.explorer.select_action(
            self.t_global.value, lambda: qout.greedy_actions.data[0],
            action_value=qout)
        q = qout.evaluate_actions(np.asarray([action]))
        self.past_action_values[self.t] = q
        self.t += 1
        self.average_q += ((1 - self.average_q_decay) *
                           (float(q.data[0]) - self.average_q))
        with self.t_global.get_lock():
            self.t_global.value += 1
            t_global = self.t_global.value

        if t_global % self.i_target == 0:
            self.logger.debug('target synchronized t_global:%s t_local:%s',
                              t_global, self.t)
            copy_param.copy_param(self.target_q_function, self.q_function)

        return action

    def act(self, obs):
        statevar = self.batch_states([obs], np, self.phi)
        qout = self.q_function(statevar)
        self.logger.debug('act action_value: %s', qout)
        return qout.greedy_actions.data[0]

    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            self.update(None)
        else:
            statevar = self.batch_states([state], np, self.phi)
            self.update(statevar)

        if isinstance(self.q_function, Recurrent):
            self.q_function.reset_state()
            self.shared_q_function.reset_state()
            self.target_q_function.reset_state()

    def stop_episode(self):
        if isinstance(self.q_function, Recurrent):
            self.q_function.reset_state()
            self.shared_q_function.reset_state()
            self.target_q_function.reset_state()

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_q_function,
                              source_link=self.q_function)

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
        ]
