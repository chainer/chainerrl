from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger
import warnings

import chainer
from chainer import functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.recurrent import RecurrentChainMixin

logger = getLogger(__name__)


class A2CModel(chainer.Link):
    """A2C model."""

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


class A2CSeparateModel(chainer.Chain, A2CModel, RecurrentChainMixin):
    """A2C model that consists of a separate policy and V-function.

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


class A2C(agent.AttributeSavingMixin, agent.BatchAgent):
    """A2C: Advantage Actor-Critic.

    A2C is a synchronous, deterministic variant of Asynchronous Advantage
        Actor Critic (A3C).

    See https://arxiv.org/abs/1708.05144

    Args:
        model (A2CModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        gamma (float): Discount factor [0,1]
        num_processes (int): The number of processes
        gpu (int): GPU device id if not None nor negative.
        update_steps (int): The number of update steps
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        entropy_coeff (float): Weight coefficient for the loss of the entropy
        use_gae (bool): use generalized advantage estimation(GAE)
        tau (float): gae parameter
        average_actor_loss_decay (float): Decay rate of average actor loss.
            Used only to record statistics.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, gamma, num_processes,
                 gpu=None,
                 update_steps=5,
                 phi=lambda x: x,
                 pi_loss_coef=1.0,
                 v_loss_coef=0.5,
                 entropy_coeff=0.01,
                 use_gae=False,
                 tau=0.95,
                 act_deterministically=False,
                 average_actor_loss_decay=0.999,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 batch_states=batch_states):

        assert isinstance(model, A2CModel)

        self.model = model
        self.gpu = gpu
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.optimizer = optimizer

        self.update_steps = update_steps
        self.num_processes = num_processes

        self.gamma = gamma
        self.use_gae = use_gae
        self.tau = tau
        self.act_deterministically = act_deterministically
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.entropy_coeff = entropy_coeff

        self.average_actor_loss_decay = average_actor_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.xp = self.model.xp
        self.t = 0
        self.t_start = 0

        # Stats
        self.average_actor_loss = 0
        self.average_value = 0
        self.average_entropy = 0

    def _flush_storage(self, obs_shape, action):
        obs_shape = obs_shape[1:]
        action_shape = action.shape[1:]

        self.states = self.xp.zeros(
            [self.update_steps + 1, self.num_processes] + list(obs_shape),
            dtype='f')
        self.actions = self.xp.zeros(
            [self.update_steps, self.num_processes] + list(action_shape),
            dtype=action.dtype)
        self.rewards = self.xp.zeros(
            (self.update_steps, self.num_processes), dtype='f')
        self.value_preds = self.xp.zeros(
            (self.update_steps + 1, self.num_processes), dtype='f')
        self.returns = self.xp.zeros(
            (self.update_steps + 1, self.num_processes), dtype='f')
        self.masks = self.xp.ones(
            (self.update_steps, self.num_processes), dtype='f')

        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def _compute_returns(self, next_value):
        if self.use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for i in reversed(range(self.update_steps)):
                delta = self.rewards[i] + \
                    self.gamma * self.value_preds[i + 1] * self.masks[i] - \
                    self.value_preds[i]
                gae = delta + self.gamma * self.tau * self.masks[i] * gae
                self.returns[i] = gae + self.value_preds[i]
        else:
            self.returns[-1] = next_value
            for i in reversed(range(self.update_steps)):
                self.returns[i] = self.rewards[i] + \
                    self.gamma * self.returns[i + 1] * self.masks[i]

    def update(self):
        with chainer.no_backprop_mode():
            _, next_value = self.model.pi_and_v(self.states[-1])
            next_value = next_value.array[:, 0]

        self._compute_returns(next_value)
        pout, values = \
            self.model.pi_and_v(chainer.Variable(
                self.states[:-1].reshape([-1] + list(self.obs_shape))))

        actions = chainer.Variable(
            self.actions.reshape([-1] + list(self.action_shape)))
        dist_entropy = F.mean(pout.entropy)
        action_log_probs = pout.log_prob(actions)

        values = values.reshape((self.update_steps, self.num_processes))
        action_log_probs = action_log_probs.reshape(
            (self.update_steps, self.num_processes))
        advantages = self.returns[:-1] - values
        value_loss = F.mean(advantages * advantages)
        action_loss = \
            - F.mean(advantages.array * action_log_probs)

        self.model.cleargrads()

        (value_loss * self.v_loss_coef +
         action_loss * self.pi_loss_coef -
         dist_entropy * self.entropy_coeff).backward()

        self.optimizer.update()
        self.states[0] = self.states[-1]

        self.t_start = self.t

        # Update stats
        self.average_actor_loss += (
            (1 - self.average_actor_loss_decay) *
            (float(action_loss.array) - self.average_actor_loss))
        self.average_value += (
            (1 - self.average_value_decay) *
            (float(value_loss.array) - self.average_value))
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(dist_entropy.array) - self.average_entropy))

    def batch_act_and_train(self, batch_obs):

        statevar = self.batch_states(batch_obs, self.xp, self.phi)

        if self.t == 0:
            with chainer.no_backprop_mode():
                pout, _ = self.model.pi_and_v(statevar)
                action = pout.sample().array
            self._flush_storage(statevar.shape, action)

        self.states[self.t - self.t_start] = statevar

        if self.t - self.t_start == self.update_steps:
            self.update()

        with chainer.no_backprop_mode():
            pout, value = self.model.pi_and_v(statevar)
            action = pout.sample().array

        self.actions[self.t - self.t_start] \
            = action.reshape([-1] + list(self.action_shape))
        self.value_preds[self.t - self.t_start] = value.array[:, 0]

        self.t += 1

        return chainer.cuda.to_cpu(action)

    def batch_act(self, batch_obs):
        statevar = self.batch_states(batch_obs, self.xp, self.phi)
        with chainer.no_backprop_mode():
            pout, _ = self.model.pi_and_v(statevar)
            action = pout.sample().array
        return chainer.cuda.to_cpu(action)

    def batch_observe_and_train(self, batch_obs, batch_reward, batch_done,
                                batch_reset):

        if any(batch_reset):
            warnings.warn('A2C currently does not support resetting an env without reaching a terminal state during training. When receiving True in batch_reset, A2C considers it as True in batch_done instead.')  # NOQA
            batch_done = list(batch_done)
            for i, reset in enumerate(batch_reset):
                if reset:
                    batch_done[i] = True

        statevar = self.batch_states(batch_obs, self.xp, self.phi)

        self.masks[self.t - self.t_start - 1] =\
            self.xp.array([0.0 if done else 1.0 for done in batch_done])
        self.rewards[self.t - self.t_start - 1] =\
            self.xp.array(batch_reward, dtype=self.xp.float32)
        self.states[self.t - self.t_start] = statevar

        if self.t - self.t_start == self.update_steps:
            self.update()

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def act_and_train(obs, reward):
        raise RuntimeError('A2C does not support non-batch training')

    def act(self, obs):
        with chainer.no_backprop_mode():
            statevar = self.batch_states([obs], self.xp, self.phi)
            pout, _ = self.model.pi_and_v(statevar)
            if self.act_deterministically:
                return chainer.cuda.to_cpu(pout.most_probable.array)[0]
            else:
                return chainer.cuda.to_cpu(pout.sample().array)[0]

    def stop_episode_and_train(self, state, reward, done=False):
        raise RuntimeError('A2C does not support non-batch training')

    def stop_episode(self):
        pass

    def get_statistics(self):
        return [
            ('average_actor', self.average_actor_loss),
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]
