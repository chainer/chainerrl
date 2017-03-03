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

    def __init__(self, capacity):
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf \S3.3
    propotional prioritization
    """

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betastep=3e-6, eps=0.0):
        # anneal beta in 200,000 steps [citation needed]
        self.alpha = alpha
        self.beta = beta0
        self.betastep = betastep
        self.eps = eps
        self.memory = PrioritizedBuffer(capacity=capacity)

    """
    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False):
    """

    def sample(self, n):
        """Sample n unique samples from this replay buffer"""
        assert len(self.memory) >= n
        sampled, probabilities = self.memory.sample(n)
        tmp = [p for p in probabilities if p is not None]
        minp = min(tmp) if len(tmp) > 0 else 1.0
        weights = [(minp if p is None else p) ** -self.beta
                   for p in probabilities]
        self.beta = min(1.0, self.beta + self.betastep)
        # return sampled, {'weights': weights}
        for e, w in zip(sampled, weights):
            e['weight'] = w
        return sampled

    def update_errors(self, errors):
        priority = [d ** self.alpha + self.eps for d in errors]
        self.memory.set_last_priority(priority)


def random_subseq(seq, subseq_len):
    if len(seq) <= subseq_len:
        return seq
    else:
        i = np.random.randint(0, len(seq) - subseq_len + 1)
        return seq[i:i + subseq_len]


class EpisodicReplayBuffer(object):

    def __init__(self, capacity):
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
            while len(self.memory) > self.capacity:
                discarded_episode = self.episodic_memory.popleft()
                for _ in range(len(discarded_episode)):
                    self.memory.popleft()
        assert not self.current_episode


class PrioritizedEpisodicReplayBuffer (EpisodicReplayBuffer):

    def __init__(self, capacity,
                 alpha=0.6, beta0=0.4, betastep=3e-6, eps=0.0):
        # anneal beta in 200,000 steps [citation needed]
        self.alpha = alpha
        self.beta = beta0
        self.betastep = betastep
        self.eps = eps

        self.current_episode = []
        self.episodic_memory = PrioritizedBuffer(capacity=None)
        self.memory = deque(maxlen=capacity)
        self.capacity_left = capacity

    """
    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, **kwargs):
    """

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer"""
        assert len(self.episodic_memory) >= n_episodes
        episodes, probabilities = self.episodic_memory.sample(n_episodes)
        tmp = [p for p in probabilities if p is not None]
        minp = min(tmp) if len(tmp) > 0 else 1.0
        weights = [(minp if p is None else p) ** -self.beta
                   for p in probabilities]
        self.beta = min(1.0, self.beta + self.betastep)
        if max_len is not None:
            episodes = [random_subseq(ep, max_len) for ep in episodes]
        for e, w in zip(episodes, weights):
            e['weight'] = w
        return episodes

    def update_errors(self, errors):
        priority = [d ** self.alpha + self.eps for d in errors]
        self.episodic_memory.set_last_priority(priority)

    def stop_current_episode(self):
        if self.current_episode:
            self.memory.extend(self.current_episode)
            self.episodic_memory.append(self.current_episode)
            self.capacity_left -= len(self.current_episode)
            self.current_episode = []
            while self.capacity_left < 0:
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


class ReplayUpdator(object):
    """Object that handles update schedule and configurations.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer
        update_func (callable): Callable that accepts one of these:
            (1) a list of transition dicts (if episodic_update=False)
            (2) a list of lists of transition dicts (if episodic_update=True)
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        batchsize (int): Minibatch size
        update_frequency (int): Model update frequency in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, replay_buffer, update_func, batchsize, episodic_update,
                 n_times_update, replay_start_size, update_frequency,
                 episodic_update_len=None):

        assert batchsize <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency

    def update_if_necessary(self, iteration):
        if len(self.replay_buffer) < self.replay_start_size:
            return
        if iteration % self.update_frequency != 0:
            return

        for _ in range(self.n_times_update):
            if self.episodic_update:
                episodes = self.replay_buffer.sample_episodes(
                    self.batchsize, self.episodic_update_len)
                self.update_func(episodes)
            else:
                transitions = self.replay_buffer.sample(self.batchsize)
                self.update_func(transitions)
