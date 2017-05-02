from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from logging import getLogger
import warnings

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
        batchsize (int): Number of episodes used for each update
        backward_separately (bool): If set true, call backward separately for
            each episode and accumulate only gradients.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        batch_states (callable): Method which makes a batch of observations.
            default is `chainerrl.misc.batch_states`
        logger (logging.Logger): Logger to be used.
    """

    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer,
                 beta=0,
                 phi=lambda x: x,
                 batchsize=1,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 backward_separately=False,
                 batch_states=chainerrl.misc.batch_states,
                 logger=None):

        self.model = model
        self.xp = self.model.xp
        self.optimizer = optimizer
        self.beta = beta
        self.phi = phi
        self.batchsize = batchsize
        self.backward_separately = backward_separately
        self.act_deterministically = act_deterministically
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states
        self.logger = logger or getLogger(__name__)

        # Statistics
        self.average_entropy = 0

        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]
        self.n_backward = 0

    def act_and_train(self, obs, reward):

        batch_obs = self.batch_states([obs], self.xp, self.phi)
        action_distrib = self.model(batch_obs)
        batch_action = action_distrib.sample().data  # Do not backprop
        action = chainer.cuda.to_cpu(batch_action)[0]

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
            batch_obs = self.batch_states([obs], self.xp, self.phi)
            action_distrib = self.model(batch_obs)
            if self.act_deterministically:
                return chainer.cuda.to_cpu(
                    action_distrib.most_probable.data)[0]
            else:
                return chainer.cuda.to_cpu(action_distrib.sample().data)[0]

    def stop_episode_and_train(self, obs, reward, done=False):

        if not done:
            warnings.warn(
                'Since REINFORCE supports episodic environments only, '
                'calling stop_episode_and_train with done=False will throw '
                'away the last episode.')
            self.reward_sequences[-1] = []
            self.log_prob_sequences[-1] = []
            self.entropy_sequences[-1] = []
        else:
            self.reward_sequences[-1].append(reward)
            if self.backward_separately:
                self.accumulate_grad()
                if self.n_backward == self.batchsize:
                    self.update_with_accumulated_grad()
            else:
                if len(self.reward_sequences) == self.batchsize:
                    self.batch_update()
                else:
                    # Prepare for the next episode
                    self.reward_sequences.append([])
                    self.log_prob_sequences.append([])
                    self.entropy_sequences.append([])

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def accumulate_grad(self):
        if self.n_backward == 0:
            self.model.zerograds()
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
        # When self.batchsize is future.types.newint.newint, dividing a
        # Variable with it will raise an error, so it is manually converted to
        # float here.
        total_loss /= float(self.batchsize)
        total_loss.backward()
        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]
        self.n_backward += 1

    def batch_update(self):
        assert len(self.reward_sequences) == self.batchsize
        assert len(self.log_prob_sequences) == self.batchsize
        assert len(self.entropy_sequences) == self.batchsize
        # Update the model
        self.model.zerograds()
        self.accumulate_grad()
        self.optimizer.update()
        self.n_backward = 0

    def update_with_accumulated_grad(self):
        assert self.n_backward == self.batchsize
        self.optimizer.update()
        self.n_backward = 0

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def get_statistics(self):
        return [
            ('average_entropy', self.average_entropy),
        ]
