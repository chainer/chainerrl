from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import copy
from logging import getLogger

import chainer
from chainer import functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc import async
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept

logger = getLogger(__name__)


class A3CModel(chainer.Link):
    """A3C model."""

    def pi_and_v(self, obs):
        """Evaluate the policy and the V-function.

        Args:
            obs (Variable or ndarray): Batched observations.
        Returns:
            Distribution and Variable
        """
        raise NotImplementedError()


class A3CSeparateModel(chainer.Chain, A3CModel, RecurrentChainMixin):
    """A3C model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        v (VFunction): V-function.
    """

    def __init__(self, pi, v):
        super().__init__(pi=pi, v=v)

    def pi_and_v(self, obs):
        pout = self.pi(obs)
        vout = self.v(obs)
        return pout, vout


class A3CSharedModel(chainer.Chain, A3CModel, RecurrentChainMixin):
    """A3C model where the policy and V-function share parameters.

    Args:
        shared (Link): Shared part. Nonlinearity must be included in it.
        pi (Policy): Policy that receives output of shared as input.
        v (VFunction): V-function that receives output of shared as input.
    """

    def __init__(self, shared, pi, v):
        super().__init__(shared=shared, pi=pi, v=v)

    def pi_and_v(self, obs):
        h = self.shared(obs)
        pout = self.pi(h)
        vout = self.v(h)
        return pout, vout


class A3C(agent.AttributeSavingMixin, agent.AsyncAgent):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783

    Args:
        model (A3CModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        process_idx (int): Index of the process.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False,
                 normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 batch_states=batch_states):

        assert isinstance(model, A3CModel)
        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        async.assert_params_not_shared(self.shared_model, self.model)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0
        # A3C won't use a explorer, but this arrtibute is referenced by run_dqn
        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    @property
    def shared_attributes(self):
        return ('shared_model', 'optimizer')

    def update(self, statevar):
        assert self.t_start < self.t

        if statevar is None:
            R = 0
        else:
            with state_kept(self.model):
                _, vout = self.model.pi_and_v(statevar)
            R = float(vout.data)

        pi_loss = 0
        v_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            if self.use_average_reward:
                R -= self.average_reward
            v = self.past_values[i]
            advantage = R - v
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * \
                    float(advantage.data)
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            # Log probability is increased proportionally to advantage
            pi_loss -= log_prob * float(advantage.data)
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function

            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and \
                self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss *= factor
            v_loss *= factor

        if self.normalize_grad_by_t_max:
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start

        if self.process_idx == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

        # Compute gradients using thread-specific model
        self.model.zerograds()
        total_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_model.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_model, source_link=self.model)
        # Update the globally shared model
        if self.process_idx == 0:
            norm = self.optimizer.compute_grads_norm()
            logger.debug('grad norm:%s', norm)
        self.optimizer.update()
        if self.process_idx == 0:
            logger.debug('update')

        self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = self.t

    def act_and_train(self, state, reward):

        statevar = self.batch_states([state], np, self.phi)

        self.past_rewards[self.t - 1] = reward

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        pout, vout = self.model.pi_and_v(statevar)
        action = pout.sample()
        action.creator = None  # Do not backprop through sampled actions
        self.past_action_log_prob[self.t] = pout.log_prob(action)
        self.past_action_entropy[self.t] = pout.entropy
        self.past_values[self.t] = vout
        self.t += 1
        action = action.data[0]
        if self.process_idx == 0:
            logger.debug('t:%s r:%s a:%s pout:%s',
                         self.t, reward, action, pout)
        # Update stats
        self.average_value += (
            (1 - self.average_value_decay) *
            (float(vout.data[0]) - self.average_value))
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(pout.entropy.data[0]) - self.average_entropy))
        return action

    def act(self, obs):
        # Use the process-local model for acting
        with chainer.no_backprop_mode():
            statevar = self.batch_states([obs], np, self.phi)
            pout, _ = self.model.pi_and_v(statevar)
            if self.act_deterministically:
                return pout.most_probable.data[0]
            else:
                return pout.sample().data[0]

    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            self.update(None)
        else:
            statevar = self.batch_states([state], np, self.phi)
            self.update(statevar)

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_model,
                              source_link=self.model)

    def get_statistics(self):
        return [
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]
