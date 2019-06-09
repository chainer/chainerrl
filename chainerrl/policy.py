from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from abc import ABCMeta
from abc import abstractmethod


from logging import getLogger
logger = getLogger(__name__)


class Policy(object, metaclass=ABCMeta):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state):
        """Evaluate a policy.

        Returns:
            Distribution of actions
        """
        raise NotImplementedError()
