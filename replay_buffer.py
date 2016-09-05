from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from future import standard_library
standard_library.install_aliases()
from collections import deque
import random


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


class EpisodicReplayBuffer(object):

    def __init__(self, capacity):
        self.current_episode = []
        self.episodic_memory = deque()
        self.memory = deque()
        self.capacity = capacity

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, new_episode=False):
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
        self.current_episode.append(experience)
        if new_episode and self.current_episode:
            self.episodic_memory.append(self.current_episode)
            self.memory.extend(self.current_episode)
            self.current_episode = []
            if len(self.memory) > self.capacity:
                discarded_episode = self.episodic_memory.popleft()
                for _ in range(discarded_episode):
                    self.memory.popleft()
                assert self.memory <= self.capacity

    def sample(self, n):
        """Sample n unique samples from this replay buffer
        """
        assert len(self.episodic_memory) >= n
        return random.sample(self.memory, n)

    def sample_episodes(self, n_episodes):
        """Sample n unique samples from this replay buffer
        """
        assert len(self.episodic_memory) >= n_episodes
        return random.sample(self.episodic_memory, n_episodes)

    def __len__(self):
        return len(self.memory)
