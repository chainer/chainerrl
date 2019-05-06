from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
from chainer import testing
from chainer.testing import condition
import numpy as np
import scipy.stats

from chainerrl import distribution


class TestSampleDiscreteActions(unittest.TestCase):

    def _test(self, gpu):
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            xp = chainer.cuda.cupy
        else:
            xp = np
        batch_probs = xp.asarray([[0.3, 0.7],
                                  [0.8, 0.2],
                                  [0.5, 0.5],
                                  [0.0, 1.0],
                                  [1.0, 0.0]], dtype=np.float32)
        counter = np.zeros(batch_probs.shape, dtype=batch_probs.dtype)
        for _ in range(1000):
            batch_indices = chainer.cuda.to_cpu(
                distribution.sample_discrete_actions(batch_probs))
            for i in range(batch_probs.shape[0]):
                counter[i][batch_indices[i]] += 1
        np.testing.assert_allclose(
            counter / 1000, chainer.cuda.to_cpu(batch_probs), atol=0.05)

    @condition.retry(3)
    def test_cpu(self):
        self._test(-1)

    @condition.retry(3)
    @testing.attr.gpu
    def test_gpu(self):
        self._test(0)


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'n': [1, 2, 10],
    'wrap_by_variable': [True, False],
    'beta': [1.0, 10.0],
    'min_prob': [0.0, 0.01, 0.1],
}))
class TestSoftmaxDistribution(unittest.TestCase):

    def setUp(self):
        self.logits = np.random.rand(self.batch_size, self.n)
        if self.wrap_by_variable:
            self.distrib = distribution.SoftmaxDistribution(
                chainer.Variable(self.logits),
                beta=self.beta,
                min_prob=self.min_prob)
        else:
            self.distrib = distribution.SoftmaxDistribution(
                self.logits,
                beta=self.beta,
                min_prob=self.min_prob)

    def test_sample(self):
        sample = self.distrib.sample()
        self.assertTrue(isinstance(sample, chainer.Variable))
        self.assertEqual(sample.shape, (self.batch_size,))
        for b in range(self.batch_size):
            self.assertGreaterEqual(sample.array[b], 0)
            self.assertLess(sample.array[b], self.n)

    def test_prob(self):
        batch_ps = []
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            self.assertTrue(isinstance(batch_p, chainer.Variable))
            for b in range(self.batch_size):
                p = batch_p.array[b]
                self.assertGreaterEqual(p, self.min_prob)
                self.assertLessEqual(p, 1)
            batch_ps.append(batch_p.array)
        np.testing.assert_almost_equal(sum(batch_ps), np.ones(self.batch_size))

    def test_log_prob(self):
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            batch_log_p = self.distrib.log_prob(indices)
            np.testing.assert_almost_equal(np.log(batch_p.array),
                                           batch_log_p.array)

    def test_entropy(self):
        self.distrib.entropy
        # TODO(fujita)

    def test_most_probable(self):
        self.distrib.most_probable
        # TODO(fujita)

    def test_self_kl(self):
        kl = self.distrib.kl(self.distrib)
        for b in range(self.batch_size):
            np.testing.assert_allclose(
                kl.array[b], np.zeros_like(kl.array[b]), rtol=1e-5)

    def test_copy(self):
        another = self.distrib.copy()
        self.assertIsNot(self.distrib, another)
        self.assertIsNot(self.distrib.logits, another.logits)


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'n': [1, 2, 10],
    'wrap_by_variable': [True, False],
}))
class TestMellowmaxDistribution(unittest.TestCase):

    def setUp(self):
        self.values = np.random.rand(self.batch_size, self.n)
        if self.wrap_by_variable:
            self.distrib = distribution.MellowmaxDistribution(
                chainer.Variable(self.values))
        else:
            self.distrib = distribution.MellowmaxDistribution(self.values)

    def test_sample(self):
        sample = self.distrib.sample()
        self.assertTrue(isinstance(sample, chainer.Variable))
        self.assertEqual(sample.shape, (self.batch_size,))
        for b in range(self.batch_size):
            self.assertGreaterEqual(sample.array[b], 0)
            self.assertLess(sample.array[b], self.n)

    def test_prob(self):
        batch_ps = []
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            self.assertTrue(isinstance(batch_p, chainer.Variable))
            for b in range(self.batch_size):
                p = batch_p.array[b]
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)
            batch_ps.append(batch_p.array)
        np.testing.assert_almost_equal(sum(batch_ps), np.ones(self.batch_size))

    def test_log_prob(self):
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            batch_log_p = self.distrib.log_prob(indices)
            np.testing.assert_almost_equal(np.log(batch_p.array),
                                           batch_log_p.array)

    def test_entropy(self):
        self.distrib.entropy
        # TODO(fujita)

    def test_most_probable(self):
        self.distrib.most_probable
        # TODO(fujita)

    def test_self_kl(self):
        kl = self.distrib.kl(self.distrib)
        for b in range(self.batch_size):
            np.testing.assert_allclose(
                kl.array[b], np.zeros_like(kl.array[b]), rtol=1e-5)

    def test_copy(self):
        another = self.distrib.copy()
        self.assertIsNot(self.distrib, another)
        self.assertIsNot(self.distrib.values, another.values)


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'ndim': [1, 2, 10],
}))
class TestGaussianDistribution(unittest.TestCase):

    def setUp(self):
        self.mean = np.random.rand(
            self.batch_size, self.ndim).astype(np.float32)
        self.var = np.random.rand(
            self.batch_size, self.ndim).astype(np.float32)
        self.distrib = distribution.GaussianDistribution(self.mean, self.var)

    def test_sample(self):
        sample = self.distrib.sample()
        self.assertTrue(isinstance(sample, chainer.Variable))
        self.assertEqual(sample.shape, (self.batch_size, self.ndim))

    def test_most_probable(self):
        most_probable = self.distrib.most_probable
        self.assertTrue(isinstance(most_probable, chainer.Variable))
        self.assertEqual(most_probable.shape, (self.batch_size, self.ndim))
        np.testing.assert_allclose(most_probable.array, self.mean, rtol=1e-5)

    def test_prob(self):
        sample = self.distrib.sample()
        sample_prob = self.distrib.prob(sample)
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_pdf = scipy.stats.multivariate_normal(
                self.mean[b], cov).pdf(sample.array[b])
            np.testing.assert_allclose(
                sample_prob.array[b],
                desired_pdf, rtol=1e-5)

    def test_log_prob(self):
        sample = self.distrib.sample()
        sample_log_prob = self.distrib.log_prob(sample)
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_pdf = scipy.stats.multivariate_normal(
                self.mean[b], cov).pdf(sample.array[b])
            np.testing.assert_allclose(
                sample_log_prob.array[b],
                np.log(desired_pdf), rtol=1e-4)

    def test_entropy(self):
        entropy = self.distrib.entropy
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_entropy = scipy.stats.multivariate_normal(
                self.mean[b], cov).entropy()
            np.testing.assert_allclose(
                entropy.array[b], desired_entropy, rtol=1e-4)

    def test_self_kl(self):
        kl = self.distrib.kl(self.distrib)
        for b in range(self.batch_size):
            np.testing.assert_allclose(
                kl.array[b], np.zeros_like(kl.array[b]), rtol=1e-5)

    def test_kl(self):
        # Compare it to chainer.functions.gaussian_kl_divergence
        standard = distribution.GaussianDistribution(
            mean=np.zeros((self.batch_size, self.ndim), dtype=np.float32),
            var=np.ones((self.batch_size, self.ndim), dtype=np.float32))
        kl = self.distrib.kl(standard)
        chainer_kl = chainer.functions.gaussian_kl_divergence(
            self.distrib.mean, self.distrib.ln_var)
        np.testing.assert_allclose(kl.array.sum(),
                                   chainer_kl.array,
                                   rtol=1e-5)

    def test_copy(self):
        another = self.distrib.copy()
        self.assertIsNot(self.distrib, another)
        self.assertIsNot(self.distrib.mean, another.mean)
        self.assertIsNot(self.distrib.var, another.var)


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'ndim': [1, 2, 10],
}))
class TestContinuousDeterministicDistribution(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(
            self.batch_size, self.ndim).astype(np.float32)
        self.distrib = distribution.ContinuousDeterministicDistribution(self.x)

    def test_sample(self):
        sample = self.distrib.sample()
        self.assertTrue(isinstance(sample, chainer.Variable))
        self.assertEqual(sample.shape, (self.batch_size, self.ndim))
        np.testing.assert_allclose(sample.array, self.x, rtol=1e-5)

    def test_most_probable(self):
        most_probable = self.distrib.most_probable
        self.assertTrue(isinstance(most_probable, chainer.Variable))
        self.assertEqual(most_probable.shape, (self.batch_size, self.ndim))
        np.testing.assert_allclose(most_probable.array, self.x, rtol=1e-5)

    def test_copy(self):
        another = self.distrib.copy()
        self.assertIsNot(self.distrib, another)
        self.assertIsNot(self.distrib.x, another.x)
