from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
from collections import deque
import os
import tempfile
import unittest

import six.moves.cPickle as pickle

from chainerrl import replay_buffer


class TestReplayBufferCompat(unittest.TestCase):

    def test_save_and_load_v0_2(self):
        capacity = 10

        tempdir = tempfile.mkdtemp()

        rbuf = ReplayBufferV0_2(capacity)

        # Add two transitions
        trans1 = dict(state=0, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans1)
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)

        # Now it has two transitions
        self.assertEqual(len(rbuf), 2)

        # Save
        filename = os.path.join(tempdir, 'rbuf.pkl')
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffer.ReplayBuffer(capacity)

        # Of course it has no transition yet
        self.assertEqual(len(rbuf), 0)

        # Load the previously saved buffer
        rbuf.load(filename)

        # Now it has two transitions again
        self.assertEqual(len(rbuf), 2)

        # And sampled transitions are exactly what I added!
        s2 = rbuf.sample(2)
        if s2[0]['state'] == 0:
            self.assertEqual(s2[0], trans1)
            self.assertEqual(s2[1], trans2)
        else:
            self.assertEqual(s2[0], trans2)
            self.assertEqual(s2[1], trans1)


class ReplayBufferV0_2(object):

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


class EpisodicReplayBufferV0_2(object):

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


class TestEpisodicReplayBufferCompat(unittest.TestCase):

    def test_save_and_load_v0_2(self):
        capacity = 10

        tempdir = tempfile.mkdtemp()

        rbuf = EpisodicReplayBufferV0_2(capacity)

        # Give clear terminals of episodes for the test because v0.2 buffer
        # didn't save episodic_memory.
        transs = [dict(state=n, action=n + 10, reward=n + 20,
                       next_state=n + 1, next_action=n + 11,
                       is_state_terminal=n in [1, 4])
                  for n in range(5)]

        # Add two episodes
        rbuf.append(**transs[0])
        rbuf.append(**transs[1])
        rbuf.stop_current_episode()

        rbuf.append(**transs[2])
        rbuf.append(**transs[3])
        rbuf.append(**transs[4])
        rbuf.stop_current_episode()

        self.assertEqual(len(rbuf), 2)  # len(rbuf) was rbuf.n_episodes

        # Save
        filename = os.path.join(tempdir, 'rbuf.pkl')
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffer.EpisodicReplayBuffer(capacity)

        # Of course it has no transition yet
        self.assertEqual(len(rbuf), 0)

        # Load the previously saved buffer
        rbuf.load(filename)

        # Sampled transitions are exactly what I added!
        s5 = rbuf.sample(5)
        self.assertEqual(len(s5), 5)
        for t in s5:
            n = t['state']
            self.assertIn(n, range(5))
            self.assertEqual(t, transs[n])

        # And sampled episodes are exactly what I added!
        s2e = rbuf.sample_episodes(2)
        self.assertEqual(len(s2e), 2)
        if s2e[0][0]['state'] == 0:
            self.assertEqual(s2e[0], [transs[0], transs[1]])
            self.assertEqual(s2e[1], [transs[2], transs[3], transs[4]])
        else:
            self.assertEqual(s2e[0], [transs[2], transs[3], transs[4]])
            self.assertEqual(s2e[1], [transs[0], transs[1]])

        # Sizes are correct!
        self.assertEqual(len(rbuf), 5)
        self.assertEqual(rbuf.n_episodes, 2)
