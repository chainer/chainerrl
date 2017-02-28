from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
import numpy as np

import chainerrl


class Boltzmann(chainerrl.explorer.Explorer):
    """Boltzmann exploration.

    Args:
        T (float): Temperature of Boltzmann distribution.
    """

    def __init__(self, T=1.0):
        self.T = T

    def select_action(self, t, greedy_action_func, action_value=None):
        assert action_value is not None
        assert isinstance(action_value,
                          chainerrl.action_value.DiscreteActionValue)
        n_actions = action_value.q_values.shape[1]
        with chainer.no_backprop_mode():
            probs = chainer.cuda.to_cpu(
                F.softmax(action_value.q_values / self.T).data).ravel()
        return np.random.choice(np.arange(n_actions),  p=probs)

    def __repr__(self):
        return 'Boltzmann(T={})'.format(self.T)
