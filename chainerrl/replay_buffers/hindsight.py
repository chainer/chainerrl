from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import collections
import copy

import numpy as np

from chainerrl.replay_buffers.episodic import EpisodicReplayBuffer  # NOQA


class HindsightReplayBuffer(EpisodicReplayBuffer):
    """Hindsight Replay Buffer

    https://arxiv.org/abs/1707.01495

    We currently do not support N-step transitions for the

    Hindsight Buffer.

    Args:
        reward_function: Takes in a state, action, and goal and returns a reward
        capacity (int): Capacity of the replay buffer
        future_k (int): number of future goals to sample per true sample
    """

    def __init__(self, reward_function,
                 capacity=None,
                 future_k=0):
        super(HindsightReplayBuffer, self).__init__(capacity)
        self.reward_function = reward_function
        # probability of sampling a future goal instead of a
        # true goal
        self.future_prob = 1.0 - 1.0/(float(future_k) + 1)

    def sample(self, n):
        assert len(self.memory) >= n
        # Select n episodes
        episodes = self.sample_episodes(n)
        # Select timesteps from each episode
        episode_lens = np.array([len(episode) for episode in episodes])
        timesteps = np.array(
            [np.random.randint(episode_lens[i]) for i in range(n)])
        # Select episodes for which we use a future goal instead of true
        her_indexes = set(
            np.where(np.random.uniform(size=n) < self.future_prob)[0])
        # Randomly select offsets of future goals
        future_offset = np.random.uniform(size=n) * (episode_lens - timesteps)
        future_offset = future_offset.astype(int)
        future_times = timesteps + future_offset
        batch = []
        # Go through episodes
        for index in range(n):
            transition = episodes[index][timesteps[index]]
            # If we are supposed to sample future goals, replace goals
            if index in her_indexes:
                transition = copy.deepcopy(transition)
                future_state = episodes[index][future_times[index]]['next_state']
                if future_state['achieved_goal'] is not None:
                    new_goal = future_state['achieved_goal']
                    transition['state']['desired_goal'] = new_goal
                    transition['next_state']['desired_goal'] = new_goal
                    transition['reward'] = self.reward_function(
                                                        transition['state'],
                                                        transition['action'],
                                                        new_goal)
            batch.append([transition])
        return batch

    def sample_episodes(self, n_episodes, max_len=None):
        episodes = self.episodic_memory.sample_with_replacement(n_episodes)
        if max_len is not None:
            return [random_subseq(ep, max_len) for ep in episodes]
        else:
            return episodes