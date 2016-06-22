from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy

import chainer
from chainer.utils import type_check


class ScaleGrad(chainer.Function):

    def __init__(self, scale):
        self.scale = scale

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

    def forward(self, x):
        return x

    def backward(self, x, gy):
        return tuple(g * self.scale for g in gy)


def scale_grad(x, scale):
    return ScaleGrad(scale=scale)(x)
