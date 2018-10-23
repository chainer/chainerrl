from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from abc import ABCMeta
from abc import abstractmethod

from future.utils import with_metaclass

from logging import getLogger
logger = getLogger(__name__)


class Policy(with_metaclass(ABCMeta, object)):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state):
        """Evaluate a policy.

        Returns:
            Distribution of actions
        """
        raise NotImplementedError()
