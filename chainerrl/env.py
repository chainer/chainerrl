from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

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
