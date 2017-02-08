from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy as np

try:
    # For Python 3.2 and later
    from functools import lru_cache
except Exception:
    from fastcache import clru_cache as lru_cache


def _get_batch_diagonal_cpu(array):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    return array[:, rows, cols]


def _set_batch_diagonal_cpu(array, diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    array[:, rows, cols] = diag_val


@lru_cache()
def _diagonal_idx_array(batch_size, n):
    idx_offsets = np.arange(
        start=0, stop=batch_size * n * n, step=n * n, dtype=np.int32).reshape(
        (batch_size, 1))
    idx = np.ravel_multi_index(
        np.diag_indices(n), (n, n)).reshape((1, n)).astype(np.int32)
    return cuda.to_gpu(idx + idx_offsets)


@lru_cache()
def _non_diagonal_idx_array(batch_size, n):
    idx_offsets = np.arange(
        start=0, stop=batch_size * n * n, step=n * n, dtype=np.int32).reshape(
        (batch_size, 1))
    idx = np.ravel_multi_index(
        np.tril_indices(n, -1), (n, n)).reshape((1, -1)).astype(np.int32)
    return cuda.to_gpu(idx + idx_offsets)


def _set_batch_diagonal_gpu(array, diag_val):
    batch_size, m, n = array.shape
    assert m == n
    cuda.cupy.ElementwiseKernel(
        'T val, int32 idx', 'raw T mat',
        'mat[idx] = val', 'lower_triangular_matrix_set_diag')(
        diag_val, _diagonal_idx_array(batch_size, n), array)


def _get_batch_diagonal_gpu(array):
    batch_size, m, n = array.shape
    assert m == n
    return cuda.cupy.ElementwiseKernel(
        'raw T mat, int32 idx', 'T val',
        'val = mat[idx]', 'lower_triangular_matrix_get_diag')(
        array, _diagonal_idx_array(batch_size, n))


def _get_batch_diagonal(array):
    xp = cuda.get_array_module(array)
    if xp == np:
        return _get_batch_diagonal_cpu(array)
    else:
        return _get_batch_diagonal_gpu(array)


def _set_batch_diagonal(array, diag_val):
    xp = cuda.get_array_module(array)
    if xp == np:
        _set_batch_diagonal_cpu(array, diag_val)
    else:
        _set_batch_diagonal_gpu(array, diag_val)


def _get_batch_non_diagonal_cpu(array):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    return array[:, rows, cols]


def _set_batch_non_diagonal_cpu(array, non_diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    array[:, rows, cols] = non_diag_val


def _set_batch_non_diagonal_gpu(array, non_diag_val):
    batch_size, m, n = array.shape
    assert m == n
    cuda.cupy.ElementwiseKernel(
        'T val, int32 idx', 'raw T mat',
        'mat[idx] = val', 'lower_triangular_matrix_set_non_diag')(
        non_diag_val, _non_diagonal_idx_array(batch_size, n), array)


def _get_batch_non_diagonal_gpu(array):
    batch_size, m, n = array.shape
    assert m == n
    return cuda.cupy.ElementwiseKernel(
        'raw T mat, int32 idx', 'T val',
        'val = mat[idx]', 'lower_triangular_matrix_get_non_diag')(
        array, _non_diagonal_idx_array(batch_size, n))


def _get_batch_non_diagonal(array):
    xp = cuda.get_array_module(array)
    if xp == np:
        return _get_batch_non_diagonal_cpu(array)
    else:
        return _get_batch_non_diagonal_gpu(array)


def _set_batch_non_diagonal(array, diag_val):
    xp = cuda.get_array_module(array)
    if xp == np:
        _set_batch_non_diagonal_cpu(array, diag_val)
    else:
        _set_batch_non_diagonal_gpu(array, diag_val)


class LowerTriangularMatrix(function.Function):
    """Compose lower triangular matrix."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2,)

    @property
    def label(self):
        return 'LowerTriangularMatrix'

    def forward(self, inputs):
        diag, non_diag = inputs
        batch_size = diag.shape[0]
        n = diag.shape[1]
        xp = cuda.get_array_module(diag)
        y = xp.zeros((batch_size, n, n), dtype=np.float32)
        _set_batch_non_diagonal(y, non_diag)
        _set_batch_diagonal(y, diag)
        return y,

    def backward(self, inputs, grad_outputs):
        diag, non_diag = inputs
        gy = grad_outputs[0]
        gdiag = _get_batch_diagonal(gy)
        gnon_diag = _get_batch_non_diagonal(gy)
        return gdiag, gnon_diag


def lower_triangular_matrix(diagonal, non_diagonal):
    return LowerTriangularMatrix()(diagonal, non_diagonal)
