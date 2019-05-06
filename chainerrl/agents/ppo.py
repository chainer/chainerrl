from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import collections
import itertools
import random

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


def _add_advantage_and_value_target_to_episode(episode, gamma, lambd):
    """Add advantage and value target values to an episode."""
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition['reward']
            + (gamma * transition['nonterminal'] * transition['next_v_pred'])
            - transition['v_pred']
        )
        adv = td_err + gamma * lambd * adv
        transition['adv'] = adv
        transition['v_teacher'] = adv + transition['v_pred']


def _add_advantage_and_value_target_to_episodes(episodes, gamma, lambd):
    """Add advantage and value target values to a list of episodes."""
    for episode in episodes:
        _add_advantage_and_value_target_to_episode(
            episode, gamma=gamma, lambd=lambd)


def _add_log_prob_and_value_to_episodes_recurrent(
        episodes,
        model,
        phi,
        batch_states,
        obs_normalizer,
):
    xp = model.xp

    # Prepare data for a recurrent model
    seqs_states = []
    seqs_next_states = []
    for ep in episodes:
        states = batch_states(
            [transition['state'] for transition in ep], xp, phi)
        next_states = batch_states(
            [transition['next_state'] for transition in ep], xp, phi)
        if obs_normalizer:
            states = obs_normalizer(states, update=False)
            next_states = obs_normalizer(next_states, update=False)
        seqs_states.append(states)
        seqs_next_states.append(next_states)

    flat_transitions = list(itertools.chain.from_iterable(episodes))

    # Predict values using a recurrent model
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        rs = model.concatenate_recurrent_states(
            [ep[0]['recurrent_state'] for ep in episodes])
        next_rs = model.concatenate_recurrent_states(
            [ep[0]['next_recurrent_state'] for ep in episodes])
        assert len(rs) == len(next_rs)

        (flat_distribs, flat_vs), _ = model.n_step_forward(
            seqs_states, recurrent_state=rs, output_mode='concat')
        (_, flat_next_vs), _ = model.n_step_forward(
            seqs_next_states, recurrent_state=next_rs, output_mode='concat')

        flat_actions = xp.array([b['action'] for b in flat_transitions])
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_log_probs = chainer.cuda.to_cpu(flat_log_probs.array)
        flat_vs = chainer.cuda.to_cpu(flat_vs.array)
        flat_next_vs = chainer.cuda.to_cpu(flat_next_vs.array)

    # Add predicted values to transitions
    for transition, log_prob, v, next_v in zip(flat_transitions,
                                               flat_log_probs,
                                               flat_vs,
                                               flat_next_vs):
        transition['log_prob'] = float(log_prob)
        transition['v_pred'] = float(v)
        transition['next_v_pred'] = float(next_v)


def _add_log_prob_and_value_to_episodes(
        episodes,
        model,
        phi,
        batch_states,
        obs_normalizer,
):

    dataset = list(itertools.chain.from_iterable(episodes))
    xp = model.xp

    # Compute v_pred and next_v_pred
    states = batch_states([b['state'] for b in dataset], xp, phi)
    next_states = batch_states([b['next_state'] for b in dataset], xp, phi)

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        distribs, vs_pred = model(states)
        _, next_vs_pred = model(next_states)

        actions = xp.array([b['action'] for b in dataset])
        log_probs = chainer.cuda.to_cpu(distribs.log_prob(actions).array)
        vs_pred = chainer.cuda.to_cpu(vs_pred.array.ravel())
        next_vs_pred = chainer.cuda.to_cpu(next_vs_pred.array.ravel())

    for transition, log_prob, v_pred, next_v_pred in zip(dataset,
                                                         log_probs,
                                                         vs_pred,
                                                         next_vs_pred):
        transition['log_prob'] = log_prob
        transition['v_pred'] = v_pred
        transition['next_v_pred'] = next_v_pred


def _limit_sequence_length(sequences, max_len):
    assert max_len > 0
    new_sequences = []
    for sequence in sequences:
        while len(sequence) > max_len:
            new_sequences.append(
                sequence[:max_len])
            sequence = sequence[max_len:]
        assert 0 < len(sequence) <= max_len
        new_sequences.append(sequence)
    return new_sequences


def _yield_subset_of_sequences_with_fixed_number_of_items(
        sequences, n_items):
    assert n_items > 0
    stack = list(reversed(sequences))
    while stack:
        subset = []
        count = 0
        while count < n_items:
            sequence = stack.pop()
            subset.append(sequence)
            count += len(sequence)
        if count > n_items:
            # Split last sequence
            sequence_to_split = subset[-1]
            n_exceeds = count - n_items
            assert n_exceeds > 0
            subset[-1] = sequence_to_split[:-n_exceeds]
            stack.append(sequence_to_split[-n_exceeds:])
        assert sum(len(seq) for seq in subset) == n_items
        yield subset


def _compute_explained_variance(transitions):
    """Compute 1 - Var[return - v]/Var[return].

    This function computes the fraction of variance that value predictions can
    explain about returns.
    """
    t = np.array([tr['v_teacher'] for tr in transitions])
    y = np.array([tr['v_pred'] for tr in transitions])
    vart = np.var(t)
    if vart == 0:
        return np.nan
    else:
        return float(1 - np.var(t - y) / vart)


def _make_dataset_recurrent(
        episodes, model, phi, batch_states, obs_normalizer,
        gamma, lambd, max_recurrent_sequence_len):
    """Make a list of sequences with necessary information."""

    _add_log_prob_and_value_to_episodes_recurrent(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
    )

    _add_advantage_and_value_target_to_episodes(
        episodes, gamma=gamma, lambd=lambd)

    if max_recurrent_sequence_len is not None:
        dataset = _limit_sequence_length(
            episodes, max_recurrent_sequence_len)
    else:
        dataset = list(episodes)

    return dataset


def _make_dataset(
        episodes, model, phi, batch_states, obs_normalizer,
        gamma, lambd):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
    )

    _add_advantage_and_value_target_to_episodes(
        episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))


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
        recurrent (bool): If set to True, `model` is assumed to implement
            `chainerrl.links.StatelessRecurrent` and update in a recurrent
            manner.
        max_recurrent_sequence_len (int): Maximum length of consecutive
            sequences of transitions in a minibatch for updatig the model.
            This value is used only when `recurrent` is True. A smaller value
            will encourage a minibatch to contain more and shorter sequences.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
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
        n_updates: Number of model updates so far.
        explained_variance: Explained variance computed from the last batch.
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
                 recurrent=False,
                 max_recurrent_sequence_len=None,
                 act_deterministically=False,
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
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.act_deterministically = act_deterministically

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

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(
            maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(
            maxlen=policy_loss_stats_window)
        self.explained_variance = np.nan

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
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                )
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory)))
            self.memory = []

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

            advs = xp.array([b['adv'] for b in batch], dtype=xp.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = xp.array([b['log_prob']
                                      for b in batch], dtype=xp.float32)
            vs_pred_old = xp.array([b['v_pred']
                                    for b in batch], dtype=xp.float32)
            vs_teacher = xp.array([b['v_teacher']
                                   for b in batch], dtype=xp.float32)
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.optimizer.update(
                self._lossfun,
                distribs.entropy, vs_pred, distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )

    def _update_once_recurrent(
            self, episodes, mean_advs, std_advs):

        assert std_advs is None or std_advs > 0

        xp = self.model.xp
        flat_transitions = list(itertools.chain.from_iterable(episodes))

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition['state'] for transition in ep], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        flat_advs = xp.array(
            [transition['adv'] for transition in flat_transitions],
            dtype=np.float32)
        if self.standardize_advantages:
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)
        flat_log_probs_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)
        flat_vs_pred_old = xp.array(
            [[transition['v_pred']] for transition in flat_transitions],
            dtype=np.float32)
        flat_vs_teacher = xp.array(
            [[transition['v_teacher']] for transition in flat_transitions],
            dtype=np.float32)

        with chainer.using_config('train', False),\
                chainer.no_backprop_mode():
            rs = self.model.concatenate_recurrent_states(
                [ep[0]['recurrent_state'] for ep in episodes])

        (flat_distribs, flat_vs_pred), _ = self.model.n_step_forward(
            seqs_states, recurrent_state=rs, output_mode='concat')
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_entropy = flat_distribs.entropy

        self.optimizer.update(
            self._lossfun,
            entropy=flat_entropy,
            vs_pred=flat_vs_pred,
            log_probs=flat_log_probs,
            vs_pred_old=flat_vs_pred_old,
            log_probs_old=flat_log_probs_old,
            advs=flat_advs,
            vs_teacher=flat_vs_teacher,
        )

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        xp = self.model.xp

        assert 'state' in flat_dataset[0]
        assert 'v_teacher' in flat_dataset[0]

        if self.standardize_advantages:
            all_advs = xp.array([b['adv'] for b in flat_dataset])
            mean_advs = xp.mean(all_advs)
            std_advs = xp.std(all_advs)
        else:
            mean_advs = None
            std_advs = None

        for epoch in range(self.epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(  # NOQA
                    dataset, self.minibatch_size):
                self._update_once_recurrent(minibatch, mean_advs, std_advs)

    def _lossfun(self,
                 entropy, vs_pred, log_probs,
                 vs_pred_old, log_probs_old,
                 advs, vs_teacher):

        prob_ratio = F.exp(log_probs - log_probs_old)

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
        loss_entropy = -F.mean(entropy)

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
            transition = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': obs,
                'nonterminal': 1.0,
            }
            if self.recurrent:
                transition['recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_prev_recurrent_states,
                        0, unwrap_variable=True)
                self.train_prev_recurrent_states = None
                transition['next_recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_recurrent_states, 0, unwrap_variable=True)
            self.last_episode.append(transition)

        self._update_if_dataset_is_ready()

        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, value = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]
            self.entropy_record.append(float(action_distrib.entropy.array))
            self.value_record.append(float(value.array))

        self.last_state = obs
        self.last_action = action

        return action

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states =\
                    self.model(b_state, self.test_recurrent_states)
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)[0]
            else:
                action = chainer.cuda.to_cpu(
                    action_distrib.sample().array)[0]

        return action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        transition = {
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': state,
            'nonterminal': 0.0 if done else 1.0,
        }
        if self.recurrent:
            transition['recurrent_state'] = self.model.get_recurrent_state_at(
                self.train_prev_recurrent_states, 0, unwrap_variable=True)
            self.train_prev_recurrent_states = None
            transition['next_recurrent_state'] =\
                self.model.get_recurrent_state_at(
                    self.train_recurrent_states, 0, unwrap_variable=True)
            self.train_recurrent_states = None
        self.last_episode.append(transition)

        self.last_state = None
        self.last_action = None

        self._flush_last_episode()
        self.stop_episode()

        self._update_if_dataset_is_ready()

    def stop_episode(self):
        self.test_recurrent_states = None

    def batch_act(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = self.model(
                    b_state, self.test_recurrent_states)
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)
            else:
                action = chainer.cuda.to_cpu(action_distrib.sample().array)

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
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, batch_value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, batch_value = self.model(b_state)
            batch_action = chainer.cuda.to_cpu(action_distrib.sample().array)
            self.entropy_record.extend(
                chainer.cuda.to_cpu(action_distrib.entropy.array))
            self.value_record.extend(chainer.cuda.to_cpu((batch_value.array)))

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.test_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.test_recurrent_states, indices_that_ended)

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
                transition = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'nonterminal': 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition['recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_prev_recurrent_states,
                            i, unwrap_variable=True)
                    transition['next_recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_recurrent_states,
                            i, unwrap_variable=True)
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.train_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.train_recurrent_states, indices_that_ended)

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ('average_value', _mean_or_nan(self.value_record)),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('average_value_loss', _mean_or_nan(self.value_loss_record)),
            ('average_policy_loss', _mean_or_nan(self.policy_loss_record)),
            ('n_updates', self.optimizer.t),
            ('explained_variance', self.explained_variance),
        ]
