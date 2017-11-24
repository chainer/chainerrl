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

    def __call__(self, obs):
        return self.pi_and_v(obs)


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


class A3C(agent.AttributeSavingMixin, agent.EpisodicActsMixin,
          agent.AsyncAgent):
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

    def update(self, state,
               past_action_log_prob,
               past_action_entropy,
               past_states,
               past_rewards,
               past_values,
               ):
        t_length = len(past_action_log_prob)
        assert t_length \
            == len(past_action_entropy) \
            == len(past_states) \
            == len(past_rewards) \
            == len(past_values)

        if state is None:
            R = 0
        else:
            with state_kept(self.model):
                _, vout = self.model.pi_and_v(state)
            R = float(vout.data)

        pi_loss = 0
        v_loss = 0
        for i in reversed(range(t_length)):
            R *= self.gamma
            R += past_rewards[i]
            if self.use_average_reward:
                R -= self.average_reward
            v = past_values[i]
            advantage = R - v
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * \
                    float(advantage.data)
            # Accumulate gradients of policy
            log_prob = past_action_log_prob[i]
            entropy = past_action_entropy[i]

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
                t_length < self.t_max:
            factor = self.t_max / t_length
            pi_loss *= factor
            v_loss *= factor

        if self.normalize_grad_by_t_max:
            pi_loss /= t_length
            v_loss /= t_length

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
            norm = sum(np.sum(np.square(param.grad))
                       for param in self.optimizer.target.params())
            logger.debug('grad norm:%s', norm)
        self.optimizer.update()
        if self.process_idx == 0:
            logger.debug('update')

        self.sync_parameters()

    def act_and_train_episode(self, state):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

        state = self.batch_states([state], np, self.phi)
        halt = False

        while not halt:
            past_action_log_prob = []
            past_action_entropy = []
            past_states = []
            past_rewards = []
            past_values = []

            while len(past_rewards) < self.t_max and not halt:
                past_states.append(state)
                pout, vout = self.model.pi_and_v(state)
                # Do not backprop through sampled actions
                action = pout.sample().data
                past_action_log_prob.append(pout.log_prob(action))
                past_action_entropy.append(pout.entropy)
                past_values.append(vout)
                action = action[0]

                # Update stats
                self.average_value += (
                    (1 - self.average_value_decay) *
                    (float(vout.data[0]) - self.average_value))
                self.average_entropy += (
                    (1 - self.average_entropy_decay) *
                    (float(pout.entropy.data[0]) - self.average_entropy))

                state, reward, halt = yield action
                past_rewards.append(reward)
                self.t += 1

                if self.process_idx == 0:
                    logger.debug('t:%s r:%s a:%s pout:%s',
                                 self.t, reward, action, pout)

                if state is not None:
                    state = self.batch_states([state], np, self.phi)

            self.update(
                state,
                past_action_log_prob,
                past_action_entropy,
                past_states,
                past_rewards,
                past_values,
            )
            if isinstance(self.model, Recurrent):
                self.model.unchain_backward()

    def act_episode(self, state):
        # Use the process-local model for acting

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

        while state is not None:
            state = self.batch_states([state], np, self.phi)

            with chainer.no_backprop_mode():
                pout, _ = self.model.pi_and_v(state)
                if self.act_deterministically:
                    state = yield pout.most_probable.data[0]
                else:
                    state = yield pout.sample().data[0]

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_model,
                              source_link=self.model)

    def get_statistics(self):
        return [
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]
