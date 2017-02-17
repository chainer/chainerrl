from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import unittest

import chainer
from chainer import testing
import numpy as np
import scipy.stats

from chainerrl import distribution


@testing.parameterize(*testing.product({
    'batch_size': [1, 3],
    'n': [1, 2, 10],
    'wrap_by_variable': [True, False],
}))
class TestSoftmaxDistribution(unittest.TestCase):

    def setUp(self):
        self.logits = np.random.rand(self.batch_size, self.n)
        if self.wrap_by_variable:
            self.distrib = distribution.SoftmaxDistribution(
                chainer.Variable(self.logits))
        else:
            self.distrib = distribution.SoftmaxDistribution(self.logits)

    def test_sample(self):
        sample = self.distrib.sample()
        self.assertTrue(isinstance(sample, chainer.Variable))
        self.assertEqual(sample.shape, (self.batch_size,))
        for b in range(self.batch_size):
            self.assertGreaterEqual(sample.data[b], 0)
            self.assertLess(sample.data[b], self.n)

    def test_prob(self):
        batch_ps = []
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            self.assertTrue(isinstance(batch_p, chainer.Variable))
            for b in range(self.batch_size):
                p = batch_p.data[b]
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)
            batch_ps.append(batch_p.data)
        np.testing.assert_almost_equal(sum(batch_ps), np.ones(self.batch_size))

    def test_log_prob(self):
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            batch_log_p = self.distrib.log_prob(indices)
            np.testing.assert_almost_equal(np.log(batch_p.data),
                                           batch_log_p.data)

    def test_entropy(self):
        self.distrib.entropy
        # TODO(fujita)

    def test_most_probable(self):
        self.distrib.most_probable
        # TODO(fujita)

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
            self.assertGreaterEqual(sample.data[b], 0)
            self.assertLess(sample.data[b], self.n)

    def test_prob(self):
        batch_ps = []
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            self.assertTrue(isinstance(batch_p, chainer.Variable))
            for b in range(self.batch_size):
                p = batch_p.data[b]
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)
            batch_ps.append(batch_p.data)
        np.testing.assert_almost_equal(sum(batch_ps), np.ones(self.batch_size))

    def test_log_prob(self):
        for a in range(self.n):
            indices = np.asarray([a] * self.batch_size, dtype=np.int32)
            batch_p = self.distrib.prob(indices)
            batch_log_p = self.distrib.log_prob(indices)
            np.testing.assert_almost_equal(np.log(batch_p.data),
                                           batch_log_p.data)

    def test_entropy(self):
        self.distrib.entropy
        # TODO(fujita)

    def test_most_probable(self):
        self.distrib.most_probable
        # TODO(fujita)

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
        np.testing.assert_allclose(most_probable.data, self.mean, rtol=1e-5)

    def test_prob(self):
        sample = self.distrib.sample()
        sample_prob = self.distrib.prob(sample)
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_pdf = scipy.stats.multivariate_normal(
                self.mean[b], cov).pdf(sample.data[b])
            np.testing.assert_allclose(
                sample_prob.data[b],
                desired_pdf, rtol=1e-5)

    def test_log_prob(self):
        sample = self.distrib.sample()
        sample_log_prob = self.distrib.log_prob(sample)
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_pdf = scipy.stats.multivariate_normal(
                self.mean[b], cov).pdf(sample.data[b])
            np.testing.assert_allclose(
                sample_log_prob.data[b],
                np.log(desired_pdf), rtol=1e-4)

    def test_entropy(self):
        entropy = self.distrib.entropy
        for b in range(self.batch_size):
            cov = (np.identity(self.ndim, dtype=np.float32) *
                   self.var[b])
            desired_entropy = scipy.stats.multivariate_normal(
                self.mean[b], cov).entropy()
            np.testing.assert_allclose(
                entropy.data[b], desired_entropy, rtol=1e-5)

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
        np.testing.assert_allclose(sample.data, self.x, rtol=1e-5)

    def test_most_probable(self):
        most_probable = self.distrib.most_probable
        self.assertTrue(isinstance(most_probable, chainer.Variable))
        self.assertEqual(most_probable.shape, (self.batch_size, self.ndim))
        np.testing.assert_allclose(most_probable.data, self.x, rtol=1e-5)

    def test_copy(self):
        another = self.distrib.copy()
        self.assertIsNot(self.distrib, another)
        self.assertIsNot(self.distrib.x, another.x)
