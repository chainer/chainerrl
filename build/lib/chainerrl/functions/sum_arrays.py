from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class SumArrays(function.Function):
    """Element-wise sum of input arrays."""

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, inputs):
        y = sum(inputs)
        return utils.force_array(y),

    def backward(self, inputs, grads):
        return [grads[0]] * len(inputs)

    def forward_gpu(self, inputs):
        n = len(inputs)
        ptrs = cuda.cupy.asarray([x.data.ptr for x in inputs],
                                 dtype=cuda.cupy.int64)
        y = cuda.elementwise(
            'T x0, int64 xs, int32 n_xs',
            'T y',
            'float** xs_ = (float**) xs;'
            'y = 0;'
            'for (size_t j = 0; j < n_xs; ++j) {'
            '  y += xs_[j][i];'
            '}',
            'sum_arrays'.format(n))(inputs[0], ptrs.data.ptr, len(ptrs))
        return y,


def sum_arrays(xs):
    """Element-wise sum of input arrays.

    Args:
        xs (tuple of ~chainer.Variable or ndarray): Input arrays to be summed.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return SumArrays()(*xs)
