from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
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


class Policy(object):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state, test=False):
        """
        Returns:
            Distribution of actions
        """
        raise NotImplementedError()


from chainerrl.policies.softmax_policy import *
from chainerrl.policies.gaussian_policy import *
from chainerrl.policies.deterministic_policy import *
