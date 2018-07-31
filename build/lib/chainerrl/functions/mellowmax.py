from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import chainer
from chainer import functions as F
import numpy as np
import scipy.optimize


def mellowmax(values, omega=1., axis=1):
    """Mellowmax function.

    This is a kind of softmax function that is, unlike the Boltzmann softmax,
    non-expansion.

    See: http://arxiv.org/abs/1612.05628

    Args:
        values (Variable or ndarray):
            Input values. Mellowmax is taken along the second axis.
        omega (float):
            Parameter of mellowmax.
        axis (int):
            Axis along which mellowmax is taken.
    Returns:
        outputs (Variable)
    """
    n = values.shape[axis]
    return (F.logsumexp(omega * values, axis=axis) - np.log(n)) / omega


def maximum_entropy_mellowmax(values, omega=1., beta_min=-10, beta_max=10):
    """Maximum entropy mellowmax policy function.

    This function provides a categorical distribution whose expectation matches
    the one of mellowmax function while maximizing its entropy.

    See: http://arxiv.org/abs/1612.05628

    Args:
        values (Variable or ndarray):
            Input values. Mellowmax is taken along the second axis.
        omega (float):
            Parameter of mellowmax.
        beta_min (float):
            Minimum value of beta, used in Brent's algorithm.
        beta_max (float):
            Maximum value of beta, used in Brent's algorithm.
    Returns:
        outputs (Variable)
    """
    xp = chainer.cuda.get_array_module(values)
    mm = mellowmax(values, axis=1)

    # Advantage: Q - mellowmax(Q)
    batch_adv = values - F.broadcast_to(F.expand_dims(mm, 1), values.shape)
    # Move data to CPU because we use Brent's algorithm in scipy
    batch_adv = chainer.cuda.to_cpu(batch_adv.data)
    batch_beta = np.empty(mm.shape, dtype=np.float32)

    # Beta is computed as the root of this function
    def f(y, adv):
        return np.sum(np.exp(y * adv) * adv)

    for idx in np.ndindex(mm.shape):
        idx_full = idx[:1] + (slice(None),) + idx[1:]
        adv = batch_adv[idx_full]
        try:
            beta = scipy.optimize.brentq(
                f, a=beta_min, b=beta_max, args=(adv,))
        except ValueError:
            beta = 0
        batch_beta[idx] = beta

    return F.softmax(xp.expand_dims(xp.asarray(batch_beta), 1) * values)
