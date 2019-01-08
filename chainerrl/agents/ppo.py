from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import collections
import copy
import itertools

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)


class PPO(agent.AttributeSavingMixin, agent.BatchAgent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (A3CModel): Model to train.  Recurrent models are not supported.
            state s  |->  (pi(s, _), v(s))
        optimizer (chainer.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
    """

    saved_attributes = ['model', 'optimizer', 'obs_normalizer']

    def __init__(self,
                 model,
                 optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1.0,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 value_loss_stats_window=100,
                 policy_loss_stats_window=100,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to_gpu(device=gpu)

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states

        self.xp = self.model.xp

        # Contains episodes used for next update iteration
        self.memory = []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(
            maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(
            maxlen=policy_loss_stats_window)

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else sum(
                len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            dataset = self._make_dataset()
            assert len(dataset) == dataset_size
            self._update(dataset)
            self.memory = []

    def _make_dataset(self):
        dataset = list(itertools.chain.from_iterable(self.memory))
        xp = self.model.xp

        # Compute v_pred and next_v_pred
        states = self.batch_states([b['state'] for b in dataset], xp, self.phi)
        next_states = self.batch_states([b['next_state']
                                         for b in dataset], xp, self.phi)
        if self.obs_normalizer:
            states = self.obs_normalizer(states, update=False)
            next_states = self.obs_normalizer(next_states, update=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            _, vs_pred = self.model(states)
            vs_pred = chainer.cuda.to_cpu(vs_pred.data.ravel())
            _, next_vs_pred = self.model(next_states)
            next_vs_pred = chainer.cuda.to_cpu(next_vs_pred.data.ravel())
        for transition, v_pred, next_v_pred in zip(dataset,
                                                   vs_pred,
                                                   next_vs_pred):
            transition['v_pred'] = v_pred
            transition['next_v_pred'] = next_v_pred

        # Compute adv and v_teacher
        for episode in self.memory:
            adv = 0.0
            for transition in reversed(episode):
                td_err = (
                    transition['reward']
                    + (self.gamma * transition['nonterminal']
                       * transition['next_v_pred'])
                    - transition['v_pred']
                )
                adv = td_err + self.gamma * self.lambd * adv
                transition['adv'] = adv
                transition['v_teacher'] = adv + transition['v_pred']

        return dataset

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = self.batch_states(
            [b['state'] for b in dataset], self.obs_normalizer.xp, self.phi)
        self.obs_normalizer.experience(states)

    def _update(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        xp = self.model.xp

        assert 'state' in dataset[0]
        assert 'v_teacher' in dataset[0]

        target_model = copy.deepcopy(self.model)

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size)

        if self.standardize_advantages:
            all_advs = xp.array([b['adv'] for b in dataset])
            mean_advs = xp.mean(all_advs)
            std_advs = xp.std(all_advs)

        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = self.batch_states(
                [b['state'] for b in batch], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = xp.array([b['action'] for b in batch])
            distribs, vs_pred = self.model(states)
            with chainer.no_backprop_mode():
                target_distribs, _ = target_model(states)
                target_log_probs = target_distribs.log_prob(actions)

            advs = xp.array([b['adv'] for b in batch], dtype=xp.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            vs_pred_old = xp.array([b['v_pred']
                                    for b in batch], dtype=xp.float32)
            vs_teacher = xp.array([b['v_teacher']
                                   for b in batch], dtype=xp.float32)
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.optimizer.update(
                self._lossfun,
                distribs, vs_pred, distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                target_log_probs=target_log_probs,
                advs=advs,
                vs_teacher=vs_teacher,
            )

    def _lossfun(self,
                 distribs, vs_pred, log_probs,
                 vs_pred_old, target_log_probs,
                 advs, vs_teacher):

        prob_ratio = F.exp(log_probs - target_log_probs)
        ent = distribs.entropy

        loss_policy = - F.mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs))

        if self.clip_eps_vf is None:
            loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        else:
            loss_value_func = F.mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_elementwise_clip(vs_pred,
                                           vs_pred_old - self.clip_eps_vf,
                                           vs_pred_old + self.clip_eps_vf)
                         - vs_teacher)
            ))
        loss_entropy = -F.mean(ent)

        self.value_loss_record.append(float(loss_value_func.array))
        self.policy_loss_record.append(float(loss_policy.array))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        return loss

    def act_and_train(self, obs, reward):

        if self.last_state is not None:
            self.last_episode.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': obs,
                'nonterminal': 1.0,
            })
        self._update_if_dataset_is_ready()

        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, value = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]
            self.entropy_record.append(float(action_distrib.entropy.data))
            self.value_record.append(float(value.data))

        self.last_state = obs
        self.last_action = action

        return action

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]

        return action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        self.last_episode.append({
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': state,
            'nonterminal': 0.0 if done else 1.0,
        })

        self.last_state = None
        self.last_action = None

        self._flush_last_episode()
        self.stop_episode()

        self._update_if_dataset_is_ready()

    def stop_episode(self):
        pass

    def batch_act(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().data)

        return action

    def batch_act_and_train(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, batch_value = self.model(b_state)
            batch_action = chainer.cuda.to_cpu(action_distrib.sample().data)
            self.entropy_record.extend(
                chainer.cuda.to_cpu(action_distrib.entropy.data))
            self.value_record.extend(chainer.cuda.to_cpu((batch_value.data)))

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):

        for i, (state, action, reward, next_state, done, reset) in enumerate(zip(  # NOQA
            self.batch_last_state,
            self.batch_last_action,
            batch_reward,
            batch_obs,
            batch_done,
            batch_reset,
        )):
            if state is not None:
                assert action is not None
                self.batch_last_episode[i].append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'nonterminal': 0.0 if done else 1.0,
                })
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
                self.batch_last_state[i] = None
                self.batch_last_action[i] = None

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ('average_value', _mean_or_nan(self.value_record)),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('average_value_loss', _mean_or_nan(self.value_loss_record)),
            ('average_policy_loss', _mean_or_nan(self.policy_loss_record)),
        ]
