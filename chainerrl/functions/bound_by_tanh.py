from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import cuda
from chainer import functions as F


def bound_by_tanh(x, low, high):
    """Bound a given value into [low, high] by tanh.

    Args:
        x (chainer.Variable): value to bound
        low (numpy.ndarray): lower bound
        high (numpy.ndarray): upper bound
    Returns: chainer.Variable
    """
    assert isinstance(x, chainer.Variable)
    assert low is not None
    assert high is not None
    xp = cuda.get_array_module(x.data)
    x_scale = (high - low) / 2
    x_scale = xp.expand_dims(xp.asarray(x_scale), axis=0)
    x_mean = (high + low) / 2
    x_mean = xp.expand_dims(xp.asarray(x_mean), axis=0)
    return F.tanh(x) * x_scale + x_mean
