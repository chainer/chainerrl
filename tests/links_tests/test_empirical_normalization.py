import unittest

import numpy as np

from chainerrl.links import empirical_normalization


class TestEmpiricalNormalization(unittest.TestCase):
    def test(self):
        en = empirical_normalization.EmpiricalNormalization(10)
        xs = []
        for _ in range(10000):
            x = np.random.normal(loc=4, scale=2, size=(7, 10))
            en(x)
            xs.extend(list(x))
        xs = np.array(xs)
        true_mean = np.mean(xs, axis=0, keepdims=True)
        true_std = np.std(xs, axis=0, keepdims=True)
        np.testing.assert_allclose(en.mean, true_mean, rtol=1e-4)
        np.testing.assert_allclose(np.sqrt(en.var), true_std, rtol=1e-4)

        x = 2 * np.random.normal(loc=4, scale=2, size=(1, 10))
        enx = en(x)
        # mean, std = en.mean_and_std()
        mean = en.mean
        std = np.sqrt(en.var)
        print('mean', mean)
        np.testing.assert_allclose(mean, np.full_like(mean, 4), rtol=1e-1)
        print('std', std)
        np.testing.assert_allclose(std, np.full_like(std, 2), rtol=1e-1)
        print('ground-truth normaliaztion', (x - 4) / 2)
        print('en(x)', enx)
        np.testing.assert_allclose((x - 4) / 2, enx, rtol=1e-1)
        np.testing.assert_allclose(x, en.inverse(enx), rtol=1e-1)
