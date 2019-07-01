from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()  # NOQA

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import collections
import copy

import numpy as np
import six.moves.cPickle as pickle

from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.collections import RandomAccessQueue
from chainerrl.misc.prioritized import PrioritizedBuffer


class AbstractReplayBuffer(with_metaclass(ABCMeta, object)):
    """Defines a common interface of replay buffer.

    You can append transitions to the replay buffer and later sample from it.
    Replay buffers are typically used in experience replay.
    """

    @abstractmethod
    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, env_id=0, **kwargs):
        """Append a transition to this replay buffer.

        Args:
            state: s_t
            action: a_t
            reward: r_t
            next_state: s_{t+1} (can be None if terminal)
            next_action: a_{t+1} (can be None for off-policy algorithms)
            is_state_terminal (bool)
            env_id (object): Object that is unique to each env. It indicates
                which env a given transition came from in multi-env training.
            **kwargs: Any other information to store.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n):
        """Sample n unique transitions from this replay buffer.

        Args:
            n (int): Number of transitions to sample.
        Returns:
            Sequence of n sampled transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the number of transitions in the buffer.

        Returns:
            Number of transitions in the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        """Save the content of the buffer to a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        """Load the content of the buffer from a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError


class AbstractEpisodicReplayBuffer(AbstractReplayBuffer):
    """Defines a common interface of episodic replay buffer.

    Episodic replay buffers allows you to append and sample episodes.
    """

    @abstractmethod
    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique (sub)episodes from this replay buffer.

        Args:
            n (int): Number of episodes to sample.
            max_len (int or None): Maximum length of sampled episodes. If it is
                smaller than the length of some episode, the subsequence of the
                episode is sampled instead. If None, full episodes are always
                returned.
        Returns:
            Sequence of n sampled episodes, each of which is a sequence of
            transitions.
        """
        raise NotImplementedError

    @abstractproperty
    def n_episodes(self):
        """Returns the number of episodes in the buffer.

        Returns:
            Number of episodes in the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_current_episode(self, env_id=0):
        """Notify the buffer that the current episode is interrupted.

        You may want to interrupt the current episode and start a new one
        before observing a terminal state. This is typical in continuing envs.
        In such cases, you need to call this method before appending a new
        transition so that the buffer will treat it as an initial transition of
        a new episode.

        This method should not be called after an episode whose termination is
        already notified by appending a transition with is_state_terminal=True.

        Args:
            env_id (object): Object that is unique to each env. It indicates
                which env's current episode is interrupted in multi-env
                training.
        """
        raise NotImplementedError


class ReplayBuffer(AbstractReplayBuffer):

    def __init__(self, capacity=None, num_steps=1):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps))

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, env_id=0, **kwargs):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))

    def stop_current_episode(self, env_id=0):
        last_n_transitions = self.last_n_transitions[env_id]
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(last_n_transitions) < self.num_steps:
            self.memory.append(list(last_n_transitions))
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del last_n_transitions[0]
        while last_n_transitions:
            self.memory.append(list(last_n_transitions))
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0

    def sample(self, num_experiences):
        assert len(self.memory) >= num_experiences
        return self.memory.sample(num_experiences)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)
        if isinstance(self.memory, collections.deque):
            # Load v0.2
            self.memory = RandomAccessQueue(
                self.memory, maxlen=self.memory.maxlen)


class PriorityWeightError(object):
    """For proportional prioritization

    alpha determines how much prioritization is used.

    beta determines how much importance sampling weights are used. beta is
    scheduled by ``beta0`` and ``betasteps``.

    Args:
        alpha (float): Exponent of errors to compute probabilities to sample
        beta0 (float): Initial value of beta
        betasteps (float): Steps to anneal beta to 1
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (str): Method to normalize weights. ``'batch'`` or
            ``True`` (default): divide by the maximum weight in the sampled
            batch. ``'memory'``: divide by the maximum weight in the memory.
            ``False``: do not normalize.
    """

    def __init__(self, alpha, beta0, betasteps, eps, normalize_by_max,
                 error_min, error_max):
        assert 0.0 <= alpha
        assert 0.0 <= beta0 <= 1.0
        self.alpha = alpha
        self.beta = beta0
        if betasteps is None:
            self.beta_add = 0
        else:
            self.beta_add = (1.0 - beta0) / betasteps
        self.eps = eps
        if normalize_by_max is True:
            normalize_by_max = 'batch'
        assert normalize_by_max in [False, 'batch', 'memory']
        self.normalize_by_max = normalize_by_max
        self.error_min = error_min
        self.error_max = error_max

    def priority_from_errors(self, errors):

        def _clip_error(error):
            if self.error_min is not None:
                error = max(self.error_min, error)
            if self.error_max is not None:
                error = min(self.error_max, error)
            return error

        return [(_clip_error(d) + self.eps) ** self.alpha for d in errors]

    def weights_from_probabilities(self, probabilities, min_probability):
        if self.normalize_by_max == 'batch':
            # discard global min and compute batch min
            min_probability = np.min(min_probability)
        if self.normalize_by_max:
            weights = [(p / min_probability) ** -self.beta
                       for p in probabilities]
        else:
            weights = [(len(self.memory) * p) ** -self.beta
                       for p in probabilities]
        self.beta = min(1.0, self.beta + self.beta_add)
        return weights


class PrioritizedReplayBuffer(ReplayBuffer, PriorityWeightError):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf Section 3.3
    proportional prioritization

    Args:
        capacity (int)
        alpha, beta0, betasteps, eps (float)
        normalize_by_max (bool)
    """

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betasteps=2e5, eps=0.01,
                 normalize_by_max=True, error_min=0,
                 error_max=1, num_steps=1):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = PrioritizedBuffer(capacity=capacity)
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps))
        PriorityWeightError.__init__(
            self, alpha, beta0, betasteps, eps, normalize_by_max,
            error_min=error_min, error_max=error_max)

    def sample(self, n):
        assert len(self.memory) >= n
        sampled, probabilities, min_prob = self.memory.sample(n)
        weights = self.weights_from_probabilities(probabilities, min_prob)
        for e, w in zip(sampled, weights):
            e[0]['weight'] = w
        return sampled

    def update_errors(self, errors):
        self.memory.set_last_priority(self.priority_from_errors(errors))


def random_subseq(seq, subseq_len):
    if len(seq) <= subseq_len:
        return seq
    else:
        i = np.random.randint(0, len(seq) - subseq_len + 1)
        return seq[i:i + subseq_len]


class EpisodicReplayBuffer(AbstractEpisodicReplayBuffer):

    def __init__(self, capacity=None):
        self.current_episode = collections.defaultdict(list)
        self.episodic_memory = RandomAccessQueue()
        self.memory = RandomAccessQueue()
        self.capacity = capacity

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, env_id=0, **kwargs):
        current_episode = self.current_episode[env_id]
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal,
                          **kwargs)
        current_episode.append(experience)
        if is_state_terminal:
            self.stop_current_episode(env_id=env_id)

    def sample(self, n):
        assert len(self.memory) >= n
        return self.memory.sample(n)

    def sample_episodes(self, n_episodes, max_len=None):
        assert len(self.episodic_memory) >= n_episodes
        episodes = self.episodic_memory.sample(n_episodes)
        if max_len is not None:
            return [random_subseq(ep, max_len) for ep in episodes]
        else:
            return episodes

    def __len__(self):
        return len(self.memory)

    @property
    def n_episodes(self):
        return len(self.episodic_memory)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.memory, self.episodic_memory), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            memory = pickle.load(f)
        if isinstance(memory, tuple):
            self.memory, self.episodic_memory = memory
        else:
            # Load v0.2
            # FIXME: The code works with EpisodicReplayBuffer
            # but not with PrioritizedEpisodicReplayBuffer
            self.memory = RandomAccessQueue(memory)
            self.episodic_memory = RandomAccessQueue()

            # Recover episodic_memory with best effort.
            episode = []
            for item in self.memory:
                episode.append(item)
                if item['is_state_terminal']:
                    self.episodic_memory.append(episode)
                    episode = []

    def stop_current_episode(self, env_id=0):
        current_episode = self.current_episode[env_id]
        if current_episode:
            self.episodic_memory.append(current_episode)
            for transition in current_episode:
                self.memory.append([transition])
            self.current_episode[env_id] = []
            while self.capacity is not None and \
                    len(self.memory) > self.capacity:
                discarded_episode = self.episodic_memory.popleft()
                for _ in range(len(discarded_episode)):
                    self.memory.popleft()
        assert not self.current_episode[env_id]


class PrioritizedEpisodicReplayBuffer (
        EpisodicReplayBuffer, PriorityWeightError):

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betasteps=2e5, eps=1e-8,
                 normalize_by_max=True,
                 default_priority_func=None,
                 uniform_ratio=0,
                 wait_priority_after_sampling=True,
                 return_sample_weights=True,
                 error_min=None,
                 error_max=None,
                 ):
        self.current_episode = collections.defaultdict(list)
        self.episodic_memory = PrioritizedBuffer(
            capacity=None,
            wait_priority_after_sampling=wait_priority_after_sampling)
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.capacity_left = capacity
        self.default_priority_func = default_priority_func
        self.uniform_ratio = uniform_ratio
        self.return_sample_weights = return_sample_weights
        PriorityWeightError.__init__(
            self, alpha, beta0, betasteps, eps, normalize_by_max,
            error_min=error_min, error_max=error_max)

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer"""
        assert len(self.episodic_memory) >= n_episodes
        episodes, probabilities, min_prob = self.episodic_memory.sample(
            n_episodes, uniform_ratio=self.uniform_ratio)
        if max_len is not None:
            episodes = [random_subseq(ep, max_len) for ep in episodes]
        if self.return_sample_weights:
            weights = self.weights_from_probabilities(probabilities, min_prob)
            return episodes, weights
        else:
            return episodes

    def update_errors(self, errors):
        self.episodic_memory.set_last_priority(
            self.priority_from_errors(errors))

    def stop_current_episode(self, env_id=0):
        current_episode = self.current_episode[env_id]
        if current_episode:
            if self.default_priority_func is not None:
                priority = self.default_priority_func(current_episode)
            else:
                priority = None
            self.memory.extend(current_episode)
            self.episodic_memory.append(current_episode, priority=priority)
            if self.capacity_left is not None:
                self.capacity_left -= len(current_episode)
            self.current_episode[env_id] = []
            while self.capacity_left is not None and self.capacity_left < 0:
                discarded_episode = self.episodic_memory.popleft()
                self.capacity_left += len(discarded_episode)
        assert not self.current_episode[env_id]


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
                future_state = episodes[index][future_times[index]]['state']
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


def batch_experiences(experiences, xp, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        xp : Numpy compatible matrix library: e.g. Numpy or CuPy.
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        'state': batch_states(
            [elem[0]['state'] for elem in experiences], xp, phi),
        'action': xp.asarray([elem[0]['action'] for elem in experiences]),
        'reward': xp.asarray([sum((gamma ** i) * exp[i]['reward']
                                  for i in range(len(exp)))
                              for exp in experiences],
                             dtype=np.float32),
        'next_state': batch_states(
            [elem[-1]['next_state']
             for elem in experiences], xp, phi),
        'is_state_terminal': xp.asarray(
            [any(transition['is_state_terminal']
                 for transition in exp) for exp in experiences],
            dtype=np.float32),
        'discount': xp.asarray([(gamma ** len(elem))for elem in experiences],
                               dtype=np.float32)}
    if all(elem[-1]['next_action'] is not None for elem in experiences):
        batch_exp['next_action'] = xp.asarray(
            [elem[-1]['next_action'] for elem in experiences])
    return batch_exp


class ReplayUpdater(object):
    """Object that handles update schedule and configurations.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer
        update_func (callable): Callable that accepts one of these:
            (1) a list of transition dicts (if episodic_update=False)
            (2) a list of lists of transition dicts (if episodic_update=True)
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        batchsize (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, replay_buffer, update_func, batchsize, episodic_update,
                 n_times_update, replay_start_size, update_interval,
                 episodic_update_len=None):

        assert batchsize <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        if len(self.replay_buffer) < self.replay_start_size:
            return

        if (self.episodic_update
                and self.replay_buffer.n_episodes < self.batchsize):
            return

        if iteration % self.update_interval != 0:
            return

        for _ in range(self.n_times_update):
            if self.episodic_update:
                episodes = self.replay_buffer.sample_episodes(
                    self.batchsize, self.episodic_update_len)
                self.update_func(episodes)
            else:
                transitions = self.replay_buffer.sample(self.batchsize)
                self.update_func(transitions)
