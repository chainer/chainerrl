from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from abc import ABCMeta
from abc import abstractmethod

from future.utils import with_metaclass


class StateQFunction(with_metaclass(ABCMeta, object)):
    """Abstract Q-function with state input."""

    @abstractmethod
    def __call__(self, x):
        """Evaluates Q-function

        Args:
            x (ndarray): state input

        Returns:
            An instance of ActionValue that allows to calculate the Q-values
            for state x and every possible action
        """
        raise NotImplementedError()


class StateActionQFunction(with_metaclass(ABCMeta, object)):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def __call__(self, x, a):
        """Evaluates Q-function

        Args:
            x (ndarray): state input
            a (ndarray): action input

        Returns:
            Q-value for state x and action a
        """
        raise NotImplementedError()
