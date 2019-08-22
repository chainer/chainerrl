from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()  # NOQA


import chainer
import numpy as np


def extract_episodes(dataset):
    """ Splits a sequential dataset of transitions into episodes.

    Args:
        dataset (chainer Dataset): a dataset consisting of sequential transitions.
    Returns:
        list of episodes, each of of which is a list of transitions
    """

    episodes = []
    current_episode = []
    for i in range(len(dataset)):
        obs, a, r, new_obs, done, info = dataset[i]
        current_episode.append((obs, a, r, new_obs, done, info))
        if done:
            episodes.append(current_episode)
            current_episode = []
    return episodes

class DemoDataset():
    """A basic demonstration dataset.

    Args:
        demos_pickle: pickle filename of demonstrations 
            (e.g. output of chainerrl.collect_demos.collect_demonstrations)
    """


    def __init__(self, demos_pickle):
        self.dataset = chainer.datasets.open_pickle_dataset(demos_pickle)

    def sample(self, n):
        """Samples transitions from the dataset

        Args:
            n (int): number of samples
        Returns:
            list of n transitions
        """
        dataset_size = len(self.dataset)
        indices = np.random.randint(dataset_size, size=n, dtype='l')
        return self.dataset[indices]


class EpisodicDemoDataset(DemoDataset):
    """A basic demonstration dataset of several episodes

    Args:
        demos_pickle: pickle filename of demonstrations 
            (e.g. output of chainerrl.collect_demos.collect_demonstrations)
    """

    def __init__(self, demos_pickle):
        DemoDataset.__init__(self, demos_pickle)
        self.episodes = extract_episodes(self.dataset)
        self.weights = [float(len(self.episodes[i])) / float(len(self.dataset))
                        for i in range(len(self.episodes))]
        np.testing.assert_almost_equal(np.sum(self.weights), 1.0) 

    def sample(self, n, trajectory_length=1):
        """Samples subtrajectories from episodes

        Args:
            n (int): number of samples
            trajectory_length (int): length of sampled trajectories
        Returns:
            list of n lists of trajectory_length transitions.
        """

        assert trajectory_length > 0
        # TODO: Sample from the episodes
        dataset_size = len(self.dataset)
        indices = np.random.randint(dataset_size, size=n, dtype='l')
        return self.dataset[indices]


class RankedDemoDataset():
    """A dataset of episodes ranked by performance quality

    Args:
        episodes: a list of lists of transitions. I.e. a list of episodes.
    """

    def __init__(self, ranked_episodes):
        self.episodes = ranked_episodes
        self.length = sum([len(episode) for episode in self.episodes])
        self.weights = [float(len(self.episodes[i])) / float(len(self))
                        for i in range(len(self.episodes))]
        np.testing.assert_almost_equal(np.sum(self.weights), 1.0) 

    def __len__(self):
        return self.length

    def sample(self, n, trajectory_length=1):
        """Samples subtrajectories from episodes

        Args:
            n (int): number of samples
            trajectory_length (int): length of sampled trajectories
        Returns:
            list of n lists of trajectory_length transitions.
        """

        assert trajectory_length > 0
        # TODO: Sample from the episodes
        return self.episodes[0][1]