from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class OPLU(function.Function):

    """Orthogonal Permuatation Linear Unit."""

    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward(self, x):
        x, = x
        xp = cuda.get_array_module(x)
        y = xp.empty_like(x, dtype=numpy.float32)
        y_former, y_latter = xp.split(y, 2, axis=self.axis)
        x_former, x_latter = xp.split(x, 2, axis=self.axis)
        xp.maximum(x_former, x_latter, out=y_former)
        xp.minimum(x_former, x_latter, out=y_latter)
        return y,

    def backward(self, x, gy):
        x, = x
        xp = cuda.get_array_module(x)
        gy, = gy
        gx = xp.empty_like(x, dtype=numpy.float32)
        gx_former, gx_latter = xp.split(gx, 2, axis=self.axis)
        gy_former, gy_latter = xp.split(gy, 2, axis=self.axis)
        x_former, x_latter = xp.split(x, 2, axis=self.axis)
        x_former_greater = x_former >= x_latter
        x_latter_greater = x_former < x_latter
        gx_former[:] = x_former_greater * gy_former + \
            x_latter_greater * gy_latter
        gx_latter[:] = x_former_greater * gy_latter + \
            x_latter_greater * gy_former
        return gx,


def oplu(x, axis=1):
    """Orthogonal Permuatation Linear Unit.

    See: http://arxiv.org/abs/1604.02313

    Args:
        x (~chainer.Variable): Input variable.
        axis (int): Axis to split into two halves so that they are paired

    Returns:
        ~chainer.Variable: Output variable.

    """
    return OPLU(axis=axis)(x)
