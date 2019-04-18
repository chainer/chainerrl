from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()  # NOQA

from abc import ABCMeta
from abc import abstractmethod


class Env(with_metaclass(ABCMeta, object)):
    """RL learning environment.

    This serves a minimal interface for RL agents.
    """

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


class VectorEnv(with_metaclass(ABCMeta, object)):
    """Parallel RL learning environments."""

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, mask):
        """Reset envs.

        Args:
            mask (Sequence of bool): Mask array that specifies which env to
                skip. If omitted, all the envs are reset.
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seeds):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            VectorEnv: The base non-wrapped VectorEnv instance
        """
        return self
