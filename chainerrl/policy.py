from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()

from logging import getLogger
logger = getLogger(__name__)

from abc import ABCMeta
from abc import abstractmethod

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

from chainerrl.links.mlp_bn import MLPBN
from chainerrl.links.mlp import MLP
from chainerrl import distribution
from chainerrl.stateful_callable import StatefulCallable


class Policy(StatefulCallable):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state, test=False):
        """
        Returns:
            Distribution of actions
        """
        raise NotImplementedError()

    def push_state(self):
        pass

    def pop_state(self):
        pass

    def reset_state(self):
        pass

    def update_state(self, x, test=False):
        """Update its state so that it reflects x and a.

        Unlike __call__, stateless QFunctions would do nothing.
        """
        pass

from chainerrl.policies.softmax_policy import *
from chainerrl.policies.gaussian_policy import *
from chainerrl.policies.deterministic_policy import *
