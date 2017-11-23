from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import os

from chainer import serializers
from future.utils import with_metaclass
import numpy
import warnings

from chainerrl.misc.makedirs import makedirs


def load_npz_no_strict(filename, obj):
    try:
        serializers.load_npz(filename, obj)
    except KeyError as e:
        warnings.warn(repr(e))
        with numpy.load(filename) as f:
            d = serializers.NpzDeserializer(f, strict=False)
            d.load(obj)


class Agent(with_metaclass(ABCMeta, object)):
    """Abstract agent class."""

    @abstractmethod
    def act_and_train(self, obs, reward):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode_and_train(self, state, reward, done=False):
        """Observe consequences and prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname):
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname):
        """Load internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_statistics(self):
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass


class EpisodicActsMixin(object):

    @abstractmethod
    def act_and_train_episode(self, observation):
        """Select actions for training.

        Arguments:
            observation: The first observation

        Receives:
            (observation, reward, halt)

        Yields:
            action
        """
        raise NotImplementedError()

    @abstractmethod
    def act_episode(self):
        """Select actions for evaluation.

        Arguments:
            observation: The first observation

        Receives:
            observation

        Yields:
            action
        """
        raise NotImplementedError()

    def act_and_train(self, obs, reward):
        try:
            session = self._ep_train_session
        except AttributeError:
            self._ep_train_session = self.act_and_train_episode(obs)
            return next(self._ep_train_session)
        else:
            # TODO(kataoka): namedtuple or OrderedDict
            return session.send((obs, reward, False))

    def act(self, obs):
        try:
            session = self._ep_session
        except AttributeError:
            self._ep_session = self.act_episode(obs)
            session = self._ep_session
            return next(session)
        else:
            return session.send(obs)

    def stop_episode_and_train(self, obs, reward, done=False):
        try:
            session = self._ep_train_session
        except AttributeError:
            pass
        else:
            if done:
                obs = None

            try:
                session.send((obs, reward, True))
            except StopIteration:
                pass
            else:
                assert False
            del self._ep_train_session

    def stop_episode(self):
        try:
            session = self._ep_session
        except AttributeError:
            pass
        else:
            try:
                session.send(None)
            except StopIteration:
                pass
            else:
                assert False
            del self._ep_session


class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self):
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname):
        """Save internal states."""
        makedirs(dirname, exist_ok=True)
        for attr in self.saved_attributes:
            serializers.save_npz(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))

    def load(self, dirname):
        """Load internal states."""
        for attr in self.saved_attributes:
            """Fix Chainer Issue #2772

            In Chainer v2, a (stateful) optimizer cannot be loaded from
            an npz saved before the first update.
            """
            load_npz_no_strict(
                os.path.join(dirname, '{}.npz'.format(attr)),
                getattr(self, attr))


class AsyncAgent(with_metaclass(ABCMeta, Agent)):
    """Abstract asynchronous agent class."""

    @abstractproperty
    def process_idx(self):
        """Index of process as integer, 0 for the representative process."""
        pass

    @abstractproperty
    def shared_attributes(self):
        """Tuple of names of shared attributes."""
        pass
