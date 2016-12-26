from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class InvertGradients(function.Function):
    """Inverts gradients of values exceeding a given range.

    See: http://arxiv.org/abs/1511.04143
    """

    def __init__(self, range_min, range_max):
        self.range_min = range_min
        self.range_max = range_max
        self.range_width = self.range_max - self.range_min
        assert (self.range_width > 0).all()

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'InvertGradients'

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        x, = inputs
        grad, = grad_outputs
        # In chainer, update will be like x.data -= lr * x.grad,
        # which means negative gradients will increase values.
        increasing = grad < 0
        grad *= ((self.range_max - x) / self.range_width * increasing +
                 (x - self.range_min) / self.range_width * (1 - increasing))
        return grad,


def invert_gradients(x, range_min, range_max):
    """Inverts gradients of values exceeding a given range.

    See: http://arxiv.org/abs/1511.04143

    Args:
        x (chainer.Variable or ndarray): Input value.
        range_min (chainer.Variable or ndarray): Minimum of the value range.
        range_max (chainer.Variable or ndarray): Maximum of the value range.
    Returns:
        The same value as x, except that the gradients backpropagated is scaled
        and inverted so that values would be in a given range after update.
    """
    xp = cuda.get_array_module(x, x.data)
    return InvertGradients(xp.asarray(range_min), xp.asarray(range_max))(x)
