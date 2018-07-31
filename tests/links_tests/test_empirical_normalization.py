import unittest

import chainer
from chainer import testing
import numpy as np

from chainerrl.links import empirical_normalization


class TestEmpiricalNormalization(unittest.TestCase):
    def test_small_cpu(self):
        self._test_small(gpu=-1)

    @testing.attr.gpu
    def test_small_gpu(self):
        self._test_small(gpu=0)

    def _test_small(self, gpu):
        en = empirical_normalization.EmpiricalNormalization(10)
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            en.to_gpu()

        xp = en.xp

        xs = []
        for t in range(10):
            x = xp.random.normal(loc=4, scale=2, size=(t + 3, 10))
            en(x)
            xs.extend(list(x))
        xs = xp.stack(xs)
        true_mean = xp.mean(xs, axis=0)
        true_std = xp.std(xs, axis=0)
        xp.testing.assert_allclose(en.mean, true_mean, rtol=1e-4)
        xp.testing.assert_allclose(en.std, true_std, rtol=1e-4)

    @testing.attr.slow
    def test_large(self):
        en = empirical_normalization.EmpiricalNormalization(10)
        for _ in range(10000):
            x = np.random.normal(loc=4, scale=2, size=(7, 10))
            en(x)
        x = 2 * np.random.normal(loc=4, scale=2, size=(1, 10))
        enx = en(x, update=False)

        np.testing.assert_allclose(en.mean, 4, rtol=1e-1)
        np.testing.assert_allclose(en.std, 2, rtol=1e-1)

        # Compare with the ground-truth normalization
        np.testing.assert_allclose((x - 4) / 2, enx, rtol=1e-1)

        # Test inverse
        np.testing.assert_allclose(x, en.inverse(enx), rtol=1e-4)

    def test_batch_axis(self):
        shape = (2, 3, 4)
        for batch_axis in range(3):
            en = empirical_normalization.EmpiricalNormalization(
                shape=shape[:batch_axis] + shape[batch_axis + 1:],
                batch_axis=batch_axis,
            )
            for _ in range(10):
                x = np.random.rand(*shape)
                en(x)

    def test_until(self):
        en = empirical_normalization.EmpiricalNormalization(7, until=20)
        last_mean = None
        last_std = None
        for t in range(15):
            en(np.random.rand(2, 7) + t)

            if 1 <= t < 10:
                self.assertFalse(np.allclose(en.mean, last_mean, rtol=1e-4))
                self.assertFalse(np.allclose(en.std, last_std, rtol=1e-4))
            elif t >= 10:
                np.testing.assert_allclose(en.mean, last_mean, rtol=1e-4)
                np.testing.assert_allclose(en.std, last_std, rtol=1e-4)

            last_mean = en.mean
            last_std = en.std

    def test_mixed_inputs(self):
        en = empirical_normalization.EmpiricalNormalization(7)
        for t in range(5):
            y = en(np.random.rand(t + 1, 7))
            self.assertIsInstance(y, np.ndarray)
            y = en(chainer.Variable(np.random.rand(t + 1, 7)))
            self.assertIsInstance(y, chainer.Variable)
