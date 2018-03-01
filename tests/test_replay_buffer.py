from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import os
import tempfile
import unittest

from chainer import testing
import numpy as np

from chainerrl import replay_buffer


class TestReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        for capacity in [100, None]:
            self.subtest_append_and_sample(capacity)

    def subtest_append_and_sample(self, capacity):
        rbuf = replay_buffer.ReplayBuffer(capacity)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        trans1 = dict(state=0, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        self.assertEqual(len(s1), 1)
        self.assertEqual(s1[0], trans1)

        # Add two and sample two, which must be unique
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), 2)
        s2 = rbuf.sample(2)
        self.assertEqual(len(s2), 2)
        if s2[0]['state'] == 0:
            self.assertEqual(s2[0], trans1)
            self.assertEqual(s2[1], trans2)
        else:
            self.assertEqual(s2[0], trans2)
            self.assertEqual(s2[1], trans1)

    def test_save_and_load(self):
        for capacity in [100, None]:
            self.subtest_append_and_sample(capacity)

    def subtest_save_and_load(self, capacity):

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.ReplayBuffer(capacity)

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


class TestEpisodicReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        for capacity in [100, None]:
            self.subtest_append_and_sample(capacity)

    def subtest_append_and_sample(self, capacity):
        rbuf = replay_buffer.EpisodicReplayBuffer(capacity)

        for n in [10, 15, 5] * 3:
            transs = [dict(state=i, action=100 + i, reward=200 + i,
                           next_state=i + 1, next_action=101 + i,
                           is_state_terminal=(i == n - 1))
                      for i in range(n)]
            for trans in transs:
                rbuf.append(**trans)

        self.assertEqual(len(rbuf), 90)
        self.assertEqual(rbuf.n_episodes, 9)

        for k in [10, 30, 90]:
            s = rbuf.sample(k)
            self.assertEqual(len(s), k)

        for k in [1, 3, 9]:
            s = rbuf.sample_episodes(k)
            self.assertEqual(len(s), k)

            s = rbuf.sample_episodes(k, max_len=10)
            for ep in s:
                self.assertLessEqual(len(ep), 10)
                for t0, t1 in zip(ep, ep[1:]):
                    self.assertEqual(t0['next_state'], t1['state'])
                    self.assertEqual(t0['next_action'], t1['action'])

    def test_save_and_load(self):
        for capacity in [100, None]:
            self.subtest_save_and_load(capacity)

    def subtest_save_and_load(self, capacity):

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.EpisodicReplayBuffer(capacity)

        transs = [dict(state=n, action=n+10, reward=n+20,
                       next_state=n+1, next_action=n+11,
                       is_state_terminal=False)
                  for n in range(5)]

        # Add two episodes
        rbuf.append(**transs[0])
        rbuf.append(**transs[1])
        rbuf.stop_current_episode()

        rbuf.append(**transs[2])
        rbuf.append(**transs[3])
        rbuf.append(**transs[4])
        rbuf.stop_current_episode()

        self.assertEqual(len(rbuf), 5)
        self.assertEqual(rbuf.n_episodes, 2)

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


class TestPrioritizedReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        for capacity in [100, None]:
            self.subtest_append_and_sample(capacity)

    def subtest_append_and_sample(self, capacity):
        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        trans1 = dict(state=0, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        rbuf.update_errors([3.14])
        self.assertEqual(len(s1), 1)
        self.assertAlmostEqual(s1[0]['weight'], 1.0)
        del s1[0]['weight']
        self.assertEqual(s1[0], trans1)

        # Add two and sample two, which must be unique
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), 2)
        s2 = rbuf.sample(2)
        rbuf.update_errors([3.14, 2.71])
        self.assertEqual(len(s2), 2)
        del s2[0]['weight']
        del s2[1]['weight']
        if s2[0]['state'] == 0:
            self.assertEqual(s2[0], trans1)
            self.assertEqual(s2[1], trans2)
        else:
            self.assertEqual(s2[0], trans2)
            self.assertEqual(s2[1], trans1)

        # Weights should be different for different TD-errors
        s3 = rbuf.sample(2)
        self.assertNotAlmostEqual(s3[0]['weight'], s3[1]['weight'])
        rbuf.update_errors([3.14, 3.14])

        # Weights should be equal for the same TD-errors
        s4 = rbuf.sample(2)
        self.assertAlmostEqual(s4[0]['weight'], s4[1]['weight'])

    def test_capacity(self):
        capacity = 10
        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity)
        # Fill the buffer
        for _ in range(capacity):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=True)
            rbuf.append(**trans1)
        self.assertEqual(len(rbuf), capacity)

        # Add a new transition
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)
        # The size should not change
        self.assertEqual(len(rbuf), capacity)

    def test_save_and_load(self):
        for capacity in [100, None]:
            self.subtest_append_and_sample(capacity)

    def subtest_save_and_load(self, capacity):

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity)

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
        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity)

        # Of course it has no transition yet
        self.assertEqual(len(rbuf), 0)

        # Load the previously saved buffer
        rbuf.load(filename)

        # Now it has two transitions again
        self.assertEqual(len(rbuf), 2)

        # And sampled transitions are exactly what I added!
        s2 = rbuf.sample(2)
        del s2[0]['weight']
        del s2[1]['weight']
        if s2[0]['state'] == 0:
            self.assertEqual(s2[0], trans1)
            self.assertEqual(s2[1], trans2)
        else:
            self.assertEqual(s2[0], trans2)
            self.assertEqual(s2[1], trans1)


def exp_return_of_episode(episode):
    return sum(np.exp(x['reward']) for x in episode)


@testing.parameterize(*(
    testing.product({
        'capacity': [100],
        'wait_priority_after_sampling': [False],
        'default_priority_func': [exp_return_of_episode],
        'uniform_ratio': [0, 0.1, 1.0],
        'return_sample_weights': [True, False],
    }) +
    testing.product({
        'capacity': [100],
        'wait_priority_after_sampling': [True],
        'default_priority_func': [None, exp_return_of_episode],
        'uniform_ratio': [0, 0.1, 1.0],
        'return_sample_weights': [True, False],
    })
))
class TestPrioritizedEpisodicReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(
            capacity=self.capacity,
            default_priority_func=self.default_priority_func,
            uniform_ratio=self.uniform_ratio,
            wait_priority_after_sampling=self.wait_priority_after_sampling,
            return_sample_weights=self.return_sample_weights)

        for n in [10, 15, 5] * 3:
            transs = [dict(state=i, action=100 + i, reward=200 + i,
                           next_state=i + 1, next_action=101 + i,
                           is_state_terminal=(i == n - 1))
                      for i in range(n)]
            for trans in transs:
                rbuf.append(**trans)

        self.assertEqual(len(rbuf), 90)
        self.assertEqual(rbuf.n_episodes, 9)

        for k in [10, 30, 90]:
            s = rbuf.sample(k)
            self.assertEqual(len(s), k)

        for k in [1, 3, 9]:
            ret = rbuf.sample_episodes(k)
            if self.return_sample_weights:
                s, wt = ret
                self.assertEqual(len(s), k)
                self.assertEqual(len(wt), k)
            else:
                s = ret
                self.assertEqual(len(s), k)
            if self.wait_priority_after_sampling:
                rbuf.update_errors([1.0] * k)

            ret = rbuf.sample_episodes(k, max_len=10)
            if self.return_sample_weights:
                s, wt = ret
                self.assertEqual(len(s), k)
                self.assertEqual(len(wt), k)
            else:
                s = ret
            if self.wait_priority_after_sampling:
                rbuf.update_errors([1.0] * k)

            for ep in s:
                self.assertLessEqual(len(ep), 10)
                for t0, t1 in zip(ep, ep[1:]):
                    self.assertEqual(t0['next_state'], t1['state'])
                    self.assertEqual(t0['next_action'], t1['action'])


class TestReplayBufferFail(unittest.TestCase):

    def setUp(self):
        self.rbuf = replay_buffer.PrioritizedReplayBuffer(100)
        self.trans1 = dict(state=0, action=1, reward=2, next_state=3,
                           next_action=4, is_state_terminal=True)
        self.rbuf.append(**self.trans1)

    def _sample1(self):
        self.rbuf.sample(1)

    def _set1(self):
        self.rbuf.update_errors([1.0])

    def test_fail_noupdate(self):
        self._sample1()
        self.assertRaises(AssertionError, self._sample1)

    def test_fail_update_first(self):
        self.assertRaises(AssertionError, self._set1)

    def test_fail_doubleupdate(self):
        self._sample1()
        self._set1()
        self.assertRaises(AssertionError, self._set1)
