from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import super
standard_library.install_aliases()

import os

import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import serializers

from . import dqn


class DDPG(dqn.DQN):
    """Deep Deterministic Policy Gradients.
    """

    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer, **kwargs):
        super().__init__(model, None, replay_buffer, gamma, explorer, **kwargs)

        # Aliases for convenience
        self.q_function = self.model['q_function']
        self.policy = self.model['policy']
        self.target_q_function = self.target_model['q_function']
        self.target_policy = self.target_model['policy']

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.average_actor_loss = 0.0
        self.average_critic_loss = 0.0

    # Update Q-function
    def compute_critic_loss(self, batch):

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']

        next_actions = self.target_policy(batch_next_state, test=True)
        next_q = self.target_q_function(batch_next_state, next_actions,
                                        test=True)

        target_q = batch_rewards + self.gamma * \
            (1.0 - batch_terminal) * next_q
        target_q.creator = None

        predict_q = self.q_function(batch_state, batch_actions, test=False)

        loss = F.mean_squared_error(target_q, predict_q)

        # Update stats
        self.average_critic_loss *= self.average_loss_decay
        self.average_critic_loss += ((1 - self.average_loss_decay) *
                                     float(loss.data))

        return loss

    def compute_actor_loss(self, batch):

        batch_state = batch['state']

        batch_size = batch_state.shape[0]

        q = self.q_function(batch_state,
                            self.policy(batch_state, test=False),
                            test=True)
        # Since we want to maximize Q, loss is negation of Q
        loss = - F.sum(q) / batch_size

        # Update stats
        self.average_actor_loss *= self.average_loss_decay
        self.average_actor_loss += ((1 - self.average_loss_decay) *
                                    float(loss.data))
        return loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences
        """

        batch = dqn.batch_experiences(experiences, self.xp, self.phi)
        self.critic_optimizer.update(lambda: self.compute_critic_loss(batch))
        self.actor_optimizer.update(lambda: self.compute_actor_loss(batch))

    def select_greedy_action(self, state):

        s = self._batch_states([state])
        action = self.policy(s, test=True)
        # Q is not needed here, but log it just for information
        q = self.q_function(s, action, test=True)

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * float(q.data)

        self.logger.debug('t:%s a:%s q:%s',
                          self.t, action.data[0], q.data)
        return cuda.to_cpu(action.data[0])

    def select_action(self, state):
        return self.explorer.select_action(
            self.t, lambda: self.select_greedy_action(state))

    def save_model(self, model_filename):
        """Save a network model to a file."""

        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt.actor',
                              self.actor_optimizer)
        serializers.save_hdf5(model_filename + '.opt.critic',
                              self.critic_optimizer)

    def load_model(self, model_filename):
        """Load a network model form a file."""

        serializers.load_hdf5(model_filename, self.model)
        self.sync_target_network()

        actor_opt_filename = model_filename + '.opt.actor'
        if os.path.exists(actor_opt_filename):
            serializers.load_hdf5(actor_opt_filename, self.actor_optimizer)
        else:
            print('WARNING: {0} was not found'.format(actor_opt_filename))

        critic_opt_filename = model_filename + '.opt.critic'
        if os.path.exists(critic_opt_filename):
            serializers.load_hdf5(critic_opt_filename, self.critic_optimizer)
        else:
            print('WARNING: {0} was not found'.format(critic_opt_filename))

    def get_stats_keys(self):
        return ('average_q', 'average_actor_loss', 'average_critic_loss')

    def get_stats_values(self):
        return (self.average_q, self.average_actor_loss, self.average_critic_loss)
