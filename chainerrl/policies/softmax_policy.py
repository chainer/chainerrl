from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *
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
from chainerrl import policy


class SoftmaxPolicy(policy.Policy):
    """Abstract softmax policy."""

    @abstractmethod
    def compute_logits(self, state):
        """
        Returns:
          ~chainer.Variable: logits of actions
        """
        raise NotImplementedError()

    def __call__(self, state):
        return distribution.SoftmaxDistribution(self.compute_logits(state))


class FCSoftmaxPolicy(chainer.ChainList, SoftmaxPolicy):
    """Softmax policy that consists of FC layers and rectifiers"""

    def __init__(self, n_input_channels, n_actions,
                 n_hidden_layers=0, n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, n_actions))
        else:
            layers.append(L.Linear(n_input_channels, n_actions))

        super(FCSoftmaxPolicy, self).__init__(*layers)

    def compute_logits(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h
