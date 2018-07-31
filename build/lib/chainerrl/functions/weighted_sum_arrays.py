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
        )

    def forward_cpu(self, inputs):
        y = sum(w * x for w, x in zip(self.weights, inputs))
        return utils.force_array(y),

    def backward(self, inputs, grads):
        return [w * grads[0] for w in self.weights]

    def forward_gpu(self, inputs):
        n = len(inputs)
        ptrs = cuda.cupy.asarray([x.data.ptr for x in inputs],
                                 dtype=cuda.cupy.int64)
        ws = cuda.cupy.asarray(self.weights, dtype=cuda.cupy.float32)
        y = cuda.elementwise(
            'T x0, int64 xs, raw W ws, int32 n_xs',
            'T y',
            'float** xs_ = (float**) xs;'
            'y = 0;'
            'for (size_t j = 0; j < n_xs; ++j) {'
            '  y += xs_[j][i] * ws[j];'
            '}',
            'weighted_sum_arrays'.format(n))(inputs[0],
                                             ptrs.data.ptr,
                                             ws,
                                             len(ptrs))
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
