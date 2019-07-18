from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy

import numpy as np

from chainerrl import replay_buffer
from chainerrl.replay_buffers.episodic import EpisodicReplayBuffer  # NOQA


class HindsightReplayBuffer(EpisodicReplayBuffer):
    """Hindsight Replay Buffer

    https://arxiv.org/abs/1707.01495

    We currently do not support N-step transitions for the

    Hindsight Buffer.

    Args:
        reward_function: (state, action, goal) -> reward
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

    def _replace_goal(self, transition, future_transition):
        transition = copy.deepcopy(transition)
        future_state = future_transition['next_state']
        assert future_state['achieved_goal'] is not None
        new_goal = future_state['achieved_goal']
        transition['state']['desired_goal'] = new_goal
        transition['next_state']['desired_goal'] = new_goal
        transition['reward'] = self.reward_function(
                                            transition['state'],
                                            transition['action'],
                                            new_goal)
        return transition

    def sample(self, n):
        assert len(self.memory) >= n
        # Select n episodes
        episodes = self.sample_episodes(n)
        # Select timesteps from each episode
        episode_lens = np.array([len(episode) for episode in episodes])
        timesteps = np.array(
            [np.random.randint(episode_lens[i]) for i in range(n)])
        # Select episodes for which we use a future goal instead of true

        do_replace = np.random.uniform(size=n) < self.future_prob
        # Randomly select offsets of future goals
        future_offset = np.random.uniform(size=n) * (episode_lens - timesteps)
        future_offset = future_offset.astype(int)
        future_times = timesteps + future_offset
        batch = []
        # Go through episodes
        for episode, timestep, future_timestep, replace in zip(
                episodes, timesteps, future_times, do_replace):
            transition = episode[timestep]
            if replace:
                future_transition = episode[future_timestep]
                transition = self._replace_goal(transition, future_transition)
            batch.append([transition])
        return batch

    def sample_episodes(self, n_episodes, max_len=None):
        episodes = self.episodic_memory.sample_with_replacement(n_episodes)
        if max_len is not None:
            return [replay_buffer.random_subseq(ep, max_len)
                    for ep in episodes]
        else:
            return episodes
