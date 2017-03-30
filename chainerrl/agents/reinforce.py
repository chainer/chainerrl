from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from logging import getLogger

import chainer
import numpy as np

import chainerrl
from chainerrl import agent
from chainerrl.recurrent import Recurrent


class REINFORCE(agent.AttributeSavingMixin, agent.Agent):
    """William's episodic REINFORCE.

    Args:
        model (Policy): Model to train. It must be a callable that accepts
            observations as input and return action distributions
            (Distribution).
        optimizer (chainer.Optimizer): optimizer used to train the model
        beta (float): Weight coefficient for the entropy regularizaiton term.
        normalize_loss_by_steps (bool): If set true, losses are normalized by
            the number of steps taken to accumulate the losses
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        logger (logging.Logger): Logger to be used.
    """

    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer,
                 beta=0,
                 phi=lambda x: x,
                 batchsize=1,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 logger=None):

        self.model = model
        self.xp = self.model.xp
        self.optimizer = optimizer
        self.beta = beta
        self.phi = phi
        self.batchsize = batchsize
        self.act_deterministically = act_deterministically
        self.average_entropy_decay = average_entropy_decay
        self.logger = logger or getLogger(__name__)

        # Statistics
        self.average_entropy = 0

        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]

    def act_and_train(self, obs, reward):

        batch_obs = self.xp.expand_dims(self.phi(obs), 0)
        action_distrib = self.model(batch_obs)
        batch_action = action_distrib.sample().data  # Do not backprop
        action = batch_action[0]

        # Save values used to compute losses
        self.reward_sequences[-1].append(reward)
        self.log_prob_sequences[-1].append(
            action_distrib.log_prob(batch_action))
        self.entropy_sequences[-1].append(
            action_distrib.entropy)

        self.t += 1

        self.logger.debug('t:%s r:%s a:%s action_distrib:%s',
                          self.t, reward, action, action_distrib)

        # Update stats
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(action_distrib.entropy.data[0]) - self.average_entropy))

        return action

    def act(self, obs):
        with chainer.no_backprop_mode():
            batch_obs = self.xp.expand_dims(self.phi(obs), 0)
            action_distrib = self.model(batch_obs)
            if self.act_deterministically:
                return action_distrib.most_probable.data[0]
            else:
                return action_distrib.sample().data[0]

    def stop_episode_and_train(self, obs, reward, done=False):

        assert done, 'REINFORCE supports episodic environments only'

        self.reward_sequences[-1].append(reward)
        if len(self.reward_sequences) == self.batchsize:
            self.update()
        else:
            self.reward_sequences.append([])
            self.log_prob_sequences.append([])
            self.entropy_sequences.append([])

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def update(self):

        assert len(self.reward_sequences) == self.batchsize
        assert len(self.log_prob_sequences) == self.batchsize
        assert len(self.entropy_sequences) == self.batchsize

        # Compute losses
        losses = []
        for r_seq, log_prob_seq, ent_seq in zip(self.reward_sequences,
                                                self.log_prob_sequences,
                                                self.entropy_sequences):
            assert len(r_seq) - 1 == len(log_prob_seq) == len(ent_seq)
            # Convert rewards into returns (=sum of future rewards)
            R_seq = np.cumsum(list(reversed(r_seq[1:])))[::-1]
            for R, log_prob, entropy in zip(R_seq, log_prob_seq, ent_seq):
                loss = -R * log_prob - self.beta * entropy
                losses.append(loss)
        total_loss = chainerrl.functions.sum_arrays(losses)
        total_loss /= self.batchsize

        # Update the model
        self.model.zerograds()
        total_loss.backward()
        self.optimizer.update()

        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def get_statistics(self):
        return [
            ('average_entropy', self.average_entropy),
        ]
