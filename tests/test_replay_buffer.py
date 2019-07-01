from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import collections
import copy
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import os
import tempfile
import unittest

from chainer import testing
import numpy as np

from chainerrl import replay_buffer


@testing.parameterize(*testing.product(
    {
        'capacity': [100, None],
        'num_steps': [1, 3]
    }
))
class TestReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffer.ReplayBuffer(capacity, num_steps)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        correct_item = collections.deque([], maxlen=num_steps)
        for i in range(num_steps):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            correct_item.append(trans1)
            rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        self.assertEqual(len(s1), 1)
        self.assertEqual(s1[0], list(correct_item))

        # Add two and sample two, which must be unique
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=False)
        correct_item2.append(trans2)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), 2)
        s2 = rbuf.sample(2)
        self.assertEqual(len(s2), 2)
        if s2[0][num_steps - 1]['state'] == 0:
            self.assertEqual(s2[0], list(correct_item))
            self.assertEqual(s2[1], list(correct_item2))
        else:
            self.assertEqual(s2[1], list(correct_item))
            self.assertEqual(s2[0], list(correct_item2))

    def test_append_and_terminate(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffer.ReplayBuffer(capacity, num_steps)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        for i in range(num_steps):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        self.assertEqual(len(s1), 1)

        # Add two and sample two, which must be unique
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), self.num_steps + 1)
        s2 = rbuf.sample(self.num_steps + 1)
        self.assertEqual(len(s2), self.num_steps + 1)
        if self.num_steps == 1:
            if s2[0][0]['state'] == 0:
                self.assertEqual(s2[1][0]['state'], 1)
            else:
                self.assertEqual(s2[1][0]['state'], 0)
        else:
            for item in s2:
                # e.g. if states are 0,0,0,1 then buffer looks like:
                # [[0,0,0], [0, 0, 1], [0, 1], [1]]
                if len(item) < self.num_steps:
                    self.assertEqual(item[len(item) - 1]['state'], 1)
                    for i in range(len(item) - 1):
                        self.assertEqual(item[i]['state'], 0)
                else:
                    for i in range(len(item) - 1):
                        self.assertEqual(item[i]['state'], 0)

    def test_stop_current_episode(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffer.ReplayBuffer(capacity, num_steps)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        for i in range(num_steps - 1):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(**trans1)
        # we haven't experienced n transitions yet
        self.assertEqual(len(rbuf), 0)
        # episode ends
        rbuf.stop_current_episode()
        # episode ends, so we should add n-1 transitions
        self.assertEqual(len(rbuf), self.num_steps - 1)

    def test_save_and_load(self):
        capacity = self.capacity
        num_steps = self.num_steps

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.ReplayBuffer(capacity, num_steps)

        correct_item = collections.deque([], maxlen=num_steps)
        # Add two transitions
        for _ in range(num_steps):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            correct_item.append(trans1)
            rbuf.append(**trans1)
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=False)
        correct_item2.append(trans2)
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
        if s2[0][num_steps - 1]['state'] == 0:
            self.assertEqual(s2[0], list(correct_item))
            self.assertEqual(s2[1], list(correct_item2))
        else:
            self.assertEqual(s2[0], list(correct_item2))
            self.assertEqual(s2[1], list(correct_item))


@testing.parameterize(*testing.product(
    {
        'capacity': [100, None],
        'future_k': [0]
    }
))
class TestHindsightReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):

        def reward_function(state, action, goal):
            return 1

        capacity = self.capacity
        future_k = self.future_k
        rbuf = replay_buffer.HindsightReplayBuffer(reward_function,
                                                   capacity,
                                                   future_k)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        trans1 = dict(state={'observation':0, 'desired_goal':1}, 
                      action=1, reward=2,
                      next_state={'observation':1, 'desired_goal':1},
                      next_action=4, is_state_terminal=True)
        correct_item = collections.deque([trans1], maxlen=1)
        rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        self.assertEqual(len(s1), 1)
        self.assertEqual(s1[0], list(correct_item))


    def test_stop_current_episode(self):
        capacity = self.capacity
        future_k = self.future_k
        def reward_function(state, action, goal):
            return 1
        rbuf = replay_buffer.HindsightReplayBuffer(reward_function,
                                                   capacity,
                                                   future_k)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        trans1 = dict(state={'observation': 0, 'desired_goal':1},
                      action=1, reward=2,
                      next_state={'observation': 3, 'desired_goal':1},
                      next_action=4, is_state_terminal=False)
        rbuf.append(**trans1)
        # episode hasn't stopped so it shouldn't be added
        self.assertEqual(len(rbuf), 0)
        # episode ends
        rbuf.stop_current_episode()
        # episode ends, so we should add the transition
        self.assertEqual(len(rbuf), 1)

    def test_save_and_load(self):
        capacity = self.capacity
        future_k = self.future_k

        def reward_function(state, action, goal):
            return 1

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.HindsightReplayBuffer(reward_function,
                                                   capacity,
                                                   future_k)

        transs = [dict(state={'observation':n, 'desired_goal':1},
                       action=n + 10, reward=n + 20,
                       next_state={'observation':n + 1, 'desired_goal':1},
                       next_action=n + 11, is_state_terminal=False)
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
        rbuf = replay_buffer.HindsightReplayBuffer(reward_function,
                                                   capacity,
                                                   future_k)


        # Of course it has no transition yet
        self.assertEqual(len(rbuf), 0)

        # Load the previously saved buffer
        rbuf.load(filename)

        # Sampled transitions are exactly what I added!
        s5 = rbuf.sample(5)
        print(s5)
        self.assertEqual(len(s5), 5)
        for t in s5:
            n = t[0]['state']['observation']
            self.assertIn(n, range(5))
            self.assertEqual(t[0], transs[n])

        # Unlike normal EpisodicBuffer, episodes
        # are sampled with replacement
        s2e = rbuf.sample_episodes(2)
        self.assertEqual(len(s2e), 2)
        episode_1 = s2e[0]
        episode_2 = s2e[1]
        episode_1_ob_1 = episode_1[0]['state']['observation']
        episode_2_ob_1 = episode_2[0]['state']['observation']
        if episode_1_ob_1 == 0:
            if episode_2_ob_1 == 0:
                # case: both are first episode
                self.assertEqual(s2e[0], [transs[0], transs[1]])
                self.assertEqual(episode_1, episode_2)
            else:
                # case: episode 1 and 2 are first and second episodes
                # respectively
                self.assertEqual(s2e[0], [transs[0], transs[1]])
                self.assertEqual(s2e[1], [transs[2], transs[3], transs[4]])
        else:
            if episode_2_ob_1 == 1:
                if episode_2_ob_1 == 0:
                    # case: episode 1 and 2 are the second and first episodes,
                    # respectively
                    self.assertEqual(s2e[1], [transs[0], transs[1]])
                    self.assertEqual(s2e[0], [transs[2], transs[3], transs[4]])                   
                else:
                    # case: both episodes are the second episode
                    self.assertEqual(s2e[0], [transs[0], transs[1]])
                    self.assertEqual(episode_1, episode_2)

        # Sizes are correct!
        self.assertEqual(len(rbuf), 5)
        self.assertEqual(rbuf.n_episodes, 2)


@testing.parameterize(*testing.product(
    {
        'capacity': [100, None],
    }
))
class TestEpisodicReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        capacity = self.capacity
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
        capacity = self.capacity

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.EpisodicReplayBuffer(capacity)

        transs = [dict(state=n, action=n + 10, reward=n + 20,
                       next_state=n + 1, next_action=n + 11,
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
            assert len(t) == 1
            n = t[0]['state']
            self.assertIn(n, range(5))
            self.assertEqual(t[0], transs[n])

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


@testing.parameterize(*testing.product(
    {
        'capacity': [100, None],
        'num_steps': [1],
        'normalize_by_max': ['batch', 'memory'],
    }
))
class TestPrioritizedReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            capacity,
            normalize_by_max=self.normalize_by_max,
            error_max=5,
            num_steps=num_steps)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        correct_item = collections.deque([], maxlen=num_steps)
        for _ in range(num_steps):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            correct_item.append(trans1)
            rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        rbuf.update_errors([3.14])
        self.assertEqual(len(s1), 1)
        self.assertAlmostEqual(s1[0][0]['weight'], 1.0)
        del s1[0][0]['weight']
        self.assertEqual(s1[0], list(correct_item))

        # Add two and sample two, which must be unique
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        correct_item2.append(trans2)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), 2)
        s2 = rbuf.sample(2)
        rbuf.update_errors([3.14, 2.71])
        self.assertEqual(len(s2), 2)
        del s2[0][0]['weight']
        del s2[1][0]['weight']
        if s2[0][num_steps - 1]['state'] == 1:
            self.assertEqual(s2[0], list(correct_item2))
            self.assertEqual(s2[1], list(correct_item))
        else:
            self.assertEqual(s2[0], list(correct_item))
            self.assertEqual(s2[1], list(correct_item2))

        # Weights should be different for different TD-errors
        s3 = rbuf.sample(2)
        self.assertNotAlmostEqual(s3[0][0]['weight'], s3[1][0]['weight'])

        # Weights should be equal for different but clipped TD-errors
        rbuf.update_errors([5, 10])
        s3 = rbuf.sample(2)
        self.assertAlmostEqual(s3[0][0]['weight'], s3[1][0]['weight'])

        # Weights should be equal for the same TD-errors
        rbuf.update_errors([3.14, 3.14])
        s4 = rbuf.sample(2)
        self.assertAlmostEqual(s4[0][0]['weight'], s4[1][0]['weight'])

    def test_capacity(self):
        capacity = self.capacity
        if capacity is None:
            return

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
        capacity = self.capacity
        num_steps = self.num_steps

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity,
                                                     num_steps=num_steps)

        # Add two transitions
        correct_item = collections.deque([], maxlen=num_steps)
        for _ in range(num_steps):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            correct_item.append(trans1)
            rbuf.append(**trans1)
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        correct_item2.append(trans2)
        rbuf.append(**trans2)

        # Now it has two transitions
        self.assertEqual(len(rbuf), 2)

        # Save
        filename = os.path.join(tempdir, 'rbuf.pkl')
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffer.PrioritizedReplayBuffer(capacity,
                                                     num_steps=num_steps)

        # Of course it has no transition yet
        self.assertEqual(len(rbuf), 0)

        # Load the previously saved buffer
        rbuf.load(filename)

        # Now it has two transitions again
        self.assertEqual(len(rbuf), 2)

        # And sampled transitions are exactly what I added!
        s2 = rbuf.sample(2)
        del s2[0][0]['weight']
        del s2[1][0]['weight']
        if s2[0][num_steps - 1]['state'] == 0:
            self.assertEqual(s2[0], list(correct_item))
            self.assertEqual(s2[1], list(correct_item2))
        else:
            self.assertEqual(s2[0], list(correct_item2))
            self.assertEqual(s2[1], list(correct_item))


def exp_return_of_episode(episode):
    return sum(np.exp(x['reward']) for x in episode)


@testing.parameterize(*(
    testing.product({
        'capacity': [100],
        'normalize_by_max': ['batch', 'memory'],
        'wait_priority_after_sampling': [False],
        'default_priority_func': [exp_return_of_episode],
        'uniform_ratio': [0, 0.1, 1.0],
        'return_sample_weights': [True, False],
    }) +
    testing.product({
        'capacity': [100],
        'normalize_by_max': ['batch', 'memory'],
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
            normalize_by_max=self.normalize_by_max,
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


@testing.parameterize(*testing.product({
    'replay_buffer_type': ['ReplayBuffer', 'PrioritizedReplayBuffer'],
}))
class TestReplayBufferWithEnvID(unittest.TestCase):

    def test(self):
        n = 5
        if self.replay_buffer_type == 'ReplayBuffer':
            rbuf = replay_buffer.ReplayBuffer(capacity=None, num_steps=n)
        elif self.replay_buffer_type == 'PrioritizedReplayBuffer':
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                capacity=None, num_steps=n)
        else:
            assert False

        # 2 transitions for env_id=0
        for i in range(2):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(env_id=0, **trans1)
        # 4 transitions for env_id=1 with a terminal state
        for i in range(4):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=(i == 3))
            rbuf.append(env_id=1, **trans1)
        # 9 transitions for env_id=2
        for i in range(9):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(env_id=2, **trans1)

        # It should have:
        #   - 4 transitions from env_id=1
        #   - 5 transitions from env_id=2
        self.assertEqual(len(rbuf), 9)

        # env_id=0 episode ends
        rbuf.stop_current_episode(env_id=0)

        # Now it should have 9 + 2 = 11 transitions
        self.assertEqual(len(rbuf), 11)

        # env_id=2 episode ends
        rbuf.stop_current_episode(env_id=2)

        # Finally it should have 9 + 2 + 4 = 15 transitions
        self.assertEqual(len(rbuf), 15)


@testing.parameterize(*testing.product({
    'replay_buffer_type': ['EpisodicReplayBuffer',
                           'PrioritizedEpisodicReplayBuffer'],
}))
class TestEpisodicReplayBufferWithEnvID(unittest.TestCase):

    def test(self):
        if self.replay_buffer_type == 'EpisodicReplayBuffer':
            rbuf = replay_buffer.EpisodicReplayBuffer(capacity=None)
        elif self.replay_buffer_type == 'PrioritizedEpisodicReplayBuffer':
            rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(capacity=None)
        else:
            assert False

        # 2 transitions for env_id=0
        for i in range(2):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(env_id=0, **trans1)
        # 4 transitions for env_id=1 with a terminal state
        for i in range(4):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=(i == 3))
            rbuf.append(env_id=1, **trans1)
        # 9 transitions for env_id=2
        for i in range(9):
            trans1 = dict(state=0, action=1, reward=2, next_state=3,
                          next_action=4, is_state_terminal=False)
            rbuf.append(env_id=2, **trans1)

        # It should have 4 transitions from env_id=1
        self.assertEqual(len(rbuf), 4)

        # env_id=0 episode ends
        rbuf.stop_current_episode(env_id=0)

        # Now it should have 4 + 2 = 6 transitions
        self.assertEqual(len(rbuf), 6)

        # env_id=2 episode ends
        rbuf.stop_current_episode(env_id=2)

        # Finally it should have 4 + 2 + 9 = 15 transitions
        self.assertEqual(len(rbuf), 15)


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


class TestBatchExperiences(unittest.TestCase):

    def test_batch_experiences(self):
        experiences = []
        experiences.append(
            [dict(state=1, action=1, reward=1, next_state=i,
                  next_action=1, is_state_terminal=False) for i in range(3)])
        experiences.append([dict(state=1, action=1, reward=1, next_state=1,
                                 next_action=1, is_state_terminal=False)])
        four_step_transition = [dict(
            state=1, action=1, reward=1, next_state=1,
            next_action=1, is_state_terminal=False)] * 3
        four_step_transition.append(dict(
                                    state=1, action=1, reward=1, next_state=5,
                                    next_action=1, is_state_terminal=True))
        experiences.append(four_step_transition)
        batch = replay_buffer.batch_experiences(
            experiences, np, lambda x: x, 0.99)
        self.assertEqual(batch['state'][0], 1)
        self.assertSequenceEqual(list(batch['is_state_terminal']),
                                 list(np.asarray([0.0, 0.0, 1.0],
                                                 dtype=np.float32)))
        self.assertSequenceEqual(list(batch['discount']),
                                 list(np.asarray([
                                      0.99 ** 3, 0.99 ** 1, 0.99 ** 4],
                                     dtype=np.float32)))
        self.assertSequenceEqual(list(batch['next_state']),
                                 list(np.asarray([2, 1, 5])))
