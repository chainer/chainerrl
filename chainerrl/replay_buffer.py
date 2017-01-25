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
        """Sample n unique samples from this replay buffer
        """
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
        self.current_episode.append(experience)
        if is_state_terminal:
            self.stop_current_episode()

    def sample(self, n):
        """Sample n unique samples from this replay buffer
        """
        assert len(self.episodic_memory) >= n
        return random.sample(self.memory, n)

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer
        """
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
