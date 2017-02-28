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

from chainerrl.misc.makedirs import makedirs
from chainerrl.misc import async


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
            serializers.load_npz(
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

    def share(self):
        """Share attributes by moving them to shared memory."""
        shared_objects = {attr: async.as_shared_objects(getattr(self, attr))
                          for attr in self.shared_attributes}
        self.set_shared_objects(shared_objects)
        return shared_objects

    def set_shared_objects(self, shared_objects):
        """Set given objects in shared memory as attributes."""
        for attr, shared in shared_objects.items():
            new_value = async.synchronize_to_shared_objects(
                getattr(self, attr), shared)
            setattr(self, attr, new_value)
