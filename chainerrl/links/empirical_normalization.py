from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
import numpy as np


class EmpiricalNormalization(chainer.Link):
    """Normalize mean and variance of values based on emprical values.

    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(self, shape, batch_axis=0, eps=1e-2, dtype=np.float32,
                 until=None, clip_threshold=None):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self._var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0
        self.register_persistent('_mean')
        self.register_persistent('_var')
        self.register_persistent('count')

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        return self.xp.squeeze(self._mean, self.batch_axis).copy()

    @property
    def std(self):
        xp = self.xp
        return xp.sqrt(xp.squeeze(self._var, self.batch_axis))

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        if isinstance(x, chainer.Variable):
            x = x.array

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        xp = self.xp

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = xp.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = xp.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var
            + delta_mean * (mean_x - self._mean)
        )

        # clear cache
        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.

        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        xp = self.xp
        mean = xp.broadcast_to(self._mean, x.shape)
        std_inv = xp.broadcast_to(self._std_inverse, x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = xp.clip(
                normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        xp = self.xp
        mean = xp.broadcast_to(self._mean, y.shape)
        std = xp.broadcast_to(xp.sqrt(self._var + self.eps), y.shape)
        return y * std + mean
