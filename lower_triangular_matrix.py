import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _batch_diagonal(array):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    return array[:, rows, cols]


def _set_batch_diagonal(array, diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    array[:, rows, cols] = diag_val


def _batch_non_diagonal(array):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    return array[:, rows, cols]


def _set_batch_non_diagonal(array, non_diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    array[:, rows, cols] = non_diag_val


class LowerTriangularMatrix(function.Function):
    """Compose lower triangular matrix."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2,)

    @property
    def label(self):
        return 'LowerTriangularMatrix'

    def forward_cpu(self, inputs):
        diag, non_diag = inputs
        batch_size = diag.shape[0]
        n = diag.shape[1]
        y = np.zeros((batch_size, n, n), dtype=np.float32)
        _set_batch_non_diagonal(y, non_diag)
        _set_batch_diagonal(y, diag)
        return y,

    def forward_gpu(self, inputs):
        # TODO(fujita) use gpu
        diag, non_diag = inputs
        diag = cuda.to_cpu(diag)
        non_diag = cuda.to_cpu(non_diag)
        y, = self.forward_cpu((diag, non_diag))
        return cuda.to_gpu(y),

    def backward_cpu(self, inputs, grad_outputs):
        diag, non_diag = inputs
        gy = grad_outputs[0]
        gdiag = _batch_diagonal(gy)
        gnon_diag = _batch_non_diagonal(gy)
        return gdiag, gnon_diag

    def backward_gpu(self, inputs, grad_outputs):
        # TODO(fujita) use gpu
        diag, non_diag = inputs
        gy = grad_outputs[0]
        diag = cuda.to_cpu(diag)
        non_diag = cuda.to_cpu(non_diag)
        gy = cuda.to_cpu(gy)
        gdiag, gnon_diag = self.backward_cpu((diag, non_diag), (gy,))
        return cuda.to_gpu(gdiag), cuda.to_gpu(gnon_diag)


def lower_triangular_matrix(diagonal, non_diagonal):
    return LowerTriangularMatrix()(diagonal, non_diagonal)
