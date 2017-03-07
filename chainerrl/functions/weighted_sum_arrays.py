from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class WeightedSumArrays(function.Function):
    """Element-wise weighted sum of input arrays."""

    def __init__(self, weights):
        self.weights = weights

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        y = sum(w * x for w, x in zip(self.weights, inputs))
        return utils.force_array(y),

    def backward(self, inputs, grads):
        return [w * grads[0] for w in self.weights]

    def forward_gpu(self, inputs):
        n = len(inputs)
        y = cuda.elementwise(
            ', '.join('T x{}'.format(i) for i in range(n)),
            'T y',
            'y = ' + '+'.join('x{} * {}'.format(i, self.weights[i])
                              for i in range(n)),
            'weighted_sum_variable_{}'.format(n))(*inputs)
        return y,


def weighted_sum_arrays(xs, weights):
    """Element-wise weighted sum of input arrays.

    Args:
        xs (tuple of ~chainer.Variable or ndarray): Input arrays to be summed.
        weights (list of float): Weight coefficients of input arrays.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return WeightedSumArrays(weights)(*xs)
