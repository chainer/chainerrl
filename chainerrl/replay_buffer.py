from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
from collections import deque
import random

import numpy as np
import six.moves.cPickle as pickle

from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.prioritized import PrioritizedBuffer


class ReplayBuffer(object):

    def __init__(self, capacity=None):
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False):
        """Append a transition to this replay buffer

        Args:
            state: s_t
            action: a_t
            reward: r_t
            next_state: s_{t+1} (can be None if terminal)
            next_action: a_{t+1} (can be None for off-policy algorithms)
            is_state_terminal (bool)
        """
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal)
        self.memory.append(experience)

    def sample(self, n):
        """Sample n unique samples from this replay buffer"""
        assert len(self.memory) >= n
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def stop_current_episode(self):
        pass


class PriorityWeightError(object):
    """For propotional prioritization

    Args:
        alpha (float): A hyperparameter that determines how much
            prioritization is used
        beta0, betasteps (float): Schedule of beta.  beta determines how much
            importance sampling weights are used.
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (bool): normalize weights by maximum priority
            of a batch.
    """

    def __init__(self, alpha, beta0, betasteps, eps, normalize_by_max):
        assert 0.0 <= alpha
        assert 0.0 <= beta0 <= 1.0
        self.alpha = alpha
        self.beta = beta0
        if betasteps is None:
            self.beta_add = 0
        else:
            self.beta_add = (1.0 - beta0) / betasteps
        self.eps = eps
        self.normalize_by_max = normalize_by_max

    def priority_from_errors(self, errors):
        return [d ** self.alpha + self.eps for d in errors]

    def weights_from_probabilities(self, probabilities):
        tmp = [p for p in probabilities if p is not None]
        minp = min(tmp) if tmp else 1.0
        probabilities = [minp if p is None else p for p in probabilities]
        if self.normalize_by_max:
            weights = [(p / minp) ** -self.beta for p in probabilities]
        else:
            weights = [(len(self.memory) * p) ** -self.beta
                       for p in probabilities]
        self.beta = min(1.0, self.beta + self.beta_add)
        return weights


class PrioritizedReplayBuffer(ReplayBuffer, PriorityWeightError):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf \S3.3
    propotional prioritization

    Args:
        capacity (int)
        alpha, beta0, betasteps, eps (float)
        normalize_by_max (bool)
    """

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betasteps=2e5, eps=1e-8,
                 normalize_by_max=True):
        self.memory = PrioritizedBuffer(capacity=capacity)
        PriorityWeightError.__init__(
            self, alpha, beta0, betasteps, eps, normalize_by_max)

    def sample(self, n):
        """Sample n unique samples from this replay buffer"""
        assert len(self.memory) >= n
        sampled, probabilities = self.memory.sample(n)
        weights = self.weights_from_probabilities(probabilities)
        for e, w in zip(sampled, weights):
            e['weight'] = w
        return sampled

    def update_errors(self, errors):
        self.memory.set_last_priority(self.priority_from_errors(errors))


def random_subseq(seq, subseq_len):
    if len(seq) <= subseq_len:
        return seq
    else:
        i = np.random.randint(0, len(seq) - subseq_len + 1)
        return seq[i:i + subseq_len]


class EpisodicReplayBuffer(object):

    def __init__(self, capacity=None):
        self.current_episode = []
        self.episodic_memory = deque()
        self.memory = deque()
        self.capacity = capacity

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, **kwargs):
        """Append a transition to this replay buffer

        Args:
            state: s_t
            action: a_t
            reward: r_t
            next_state: s_{t+1} (can be None if terminal)
            next_action: a_{t+1} (can be None for off-policy algorithms)
            is_state_terminal (bool)
        """
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal,
                          **kwargs)
        self.current_episode.append(experience)
        if is_state_terminal:
            self.stop_current_episode()

    def sample(self, n):
        """Sample n unique samples from this replay buffer"""
        assert len(self.memory) >= n
        return random.sample(self.memory, n)

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer"""
        assert len(self.episodic_memory) >= n_episodes
        episodes = random.sample(self.episodic_memory, n_episodes)
        if max_len is not None:
            return [random_subseq(ep, max_len) for ep in episodes]
        else:
            return episodes

    def __len__(self):
        return len(self.episodic_memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def stop_current_episode(self):
        if self.current_episode:
            self.episodic_memory.append(self.current_episode)
            self.memory.extend(self.current_episode)
            self.current_episode = []
            while self.capacity is not None and \
                    len(self.memory) > self.capacity:
                discarded_episode = self.episodic_memory.popleft()
                for _ in range(len(discarded_episode)):
                    self.memory.popleft()
        assert not self.current_episode


class PrioritizedEpisodicReplayBuffer (
        EpisodicReplayBuffer, PriorityWeightError):

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betasteps=2e5, eps=1e-8,
                 normalize_by_max=True,
                 default_priority_func=None,
                 uniform_ratio=0,
                 wait_priority_after_sampling=True,
                 return_sample_weights=True):
        self.current_episode = []
        self.episodic_memory = PrioritizedBuffer(
            capacity=None,
            wait_priority_after_sampling=wait_priority_after_sampling)
        self.memory = deque(maxlen=capacity)
        self.capacity_left = capacity
        self.default_priority_func = default_priority_func
        self.uniform_ratio = uniform_ratio
        self.return_sample_weights = return_sample_weights
        PriorityWeightError.__init__(
            self, alpha, beta0, betasteps, eps, normalize_by_max)

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer"""
        assert len(self.episodic_memory) >= n_episodes
        episodes, probabilities = self.episodic_memory.sample(
            n_episodes, uniform_ratio=self.uniform_ratio)
        if max_len is not None:
            episodes = [random_subseq(ep, max_len) for ep in episodes]
        if self.return_sample_weights:
            weights = self.weights_from_probabilities(probabilities)
            return episodes, weights
        else:
            return episodes

    def update_errors(self, errors):
        self.episodic_memory.set_last_priority(
            self.priority_from_errors(errors))

    def stop_current_episode(self):
        if self.current_episode:
            if self.default_priority_func is not None:
                priority = self.default_priority_func(self.current_episode)
            else:
                priority = None
            self.memory.extend(self.current_episode)
            self.episodic_memory.append(self.current_episode,
                                        priority=priority)
            if self.capacity_left is not None:
                self.capacity_left -= len(self.current_episode)
            self.current_episode = []
            while self.capacity_left is not None and self.capacity_left < 0:
                discarded_episode = self.episodic_memory.pop()
                self.capacity_left += len(discarded_episode)
        assert not self.current_episode


def batch_experiences(experiences, xp, phi, batch_states=batch_states):

    return {
        'state': batch_states(
            [elem['state'] for elem in experiences], xp, phi),
        'action': xp.asarray([elem['action'] for elem in experiences]),
        'reward': xp.asarray(
            [elem['reward'] for elem in experiences], dtype=np.float32),
        'next_state': batch_states(
            [elem['next_state'] for elem in experiences], xp, phi),
        'next_action': xp.asarray(
            [elem['next_action'] for elem in experiences]),
        'is_state_terminal': xp.asarray(
            [elem['is_state_terminal'] for elem in experiences],

            dtype=np.float32)}


class ReplayUpdater(object):
    """Object that handles update schedule and configurations.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer
        update_func (callable): Callable that accepts one of these:
            (1) a list of transition dicts (if episodic_update=False)
            (2) a list of lists of transition dicts (if episodic_update=True)
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        batchsize (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, replay_buffer, update_func, batchsize, episodic_update,
                 n_times_update, replay_start_size, update_interval,
                 episodic_update_len=None):

        assert batchsize <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        if len(self.replay_buffer) < self.replay_start_size:
            return
        if iteration % self.update_interval != 0:
            return

        for _ in range(self.n_times_update):
            if self.episodic_update:
                episodes = self.replay_buffer.sample_episodes(
                    self.batchsize, self.episodic_update_len)
                self.update_func(episodes)
            else:
                transitions = self.replay_buffer.sample(self.batchsize)
                self.update_func(transitions)
