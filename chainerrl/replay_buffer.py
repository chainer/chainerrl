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
        assert len(self.episodic_memory) >= n
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
        return len(self.memory)

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
