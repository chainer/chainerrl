from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class SumArrays(function.Function):
    """Element-wise sum of input arrays."""

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        y = sum(inputs)
        return utils.force_array(y),

    def backward(self, inputs, grads):
        return [grads[0]] * len(inputs)

    def forward_gpu(self, inputs):
        n = len(inputs)
        y = cuda.elementwise(
            ', '.join('T x{}'.format(i) for i in range(n)),
            'T y',
            'y = ' + '+'.join('x{}'.format(i) for i in range(n)),
            'sum_variable_{}'.format(n))(*inputs)
        return y,


def sum_arrays(xs):
    """Element-wise sum of input arrays.

    Args:
        xs (tuple of ~chainer.Variable or ndarray): Input arrays to be summed.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return SumArrays()(*xs)
