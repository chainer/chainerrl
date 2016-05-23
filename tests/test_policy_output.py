import unittest

import numpy as np
import chainer
import scipy.stats
from chainer import functions as F

from policy_output import GaussianPolicyOutput


class _TestGaussianPolicyOutput(unittest.TestCase):

    def test_sampled_actions_log_probs(self):
        sampled_actions = self.pout.sampled_actions
        print('sampled_actions', sampled_actions.data)
        for i in range(self.shape[0]):
            cov = np.identity(
                self.shape[1], dtype=np.float32) * np.exp(self.pout.ln_var.data[i])
            print('cov', cov)
            desired_pdf = scipy.stats.multivariate_normal(
                self.pout.mean.data[i], cov).pdf(sampled_actions.data[i]).astype(np.float32)
            np.testing.assert_allclose(
                self.pout.sampled_actions_log_probs.data[i], np.log(desired_pdf), rtol=1e-5)

    def test_entropy(self):
        pout = GaussianPolicyOutput(self.pout.mean, self.pout.ln_var)
        for i in range(self.shape[0]):
            cov = np.identity(
                self.shape[1], dtype=np.float32) * np.exp(self.pout.ln_var.data[i])
            print('cov', cov)
            desired_entropy = scipy.stats.multivariate_normal(
                self.pout.mean.data[i], cov).entropy()
            np.testing.assert_allclose(
                pout.entropy.data[i], desired_entropy.astype(np.float32), rtol=1e-5)


class TestGaussianPolicyOutputWithDiagonalVariance(_TestGaussianPolicyOutput):

    def setUp(self):
        self.shape = (2, 3)
        mean = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('mean', mean.data)
        var = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('var', var.data)
        self.pout = GaussianPolicyOutput(mean, var=var)


class TestGaussianPolicyOutputWithDiagonalLogVariance(_TestGaussianPolicyOutput):

    def setUp(self):
        self.shape = (2, 3)
        mean = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('mean', mean.data)
        # Log of variance can be negative
        ln_var = chainer.Variable(
            np.random.uniform(-1.0, 1.0, self.shape).astype(np.float32))
        print('ln_var', ln_var.data)
        self.pout = GaussianPolicyOutput(mean, ln_var=ln_var)


class TestGaussianPolicyOutputEquivalence(unittest.TestCase):

    def test_equivalence(self):
        self.shape = (2, 3)
        mean = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('mean', mean.data)
        var = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('var', var.data)
        ln_var = F.log(var)
        print('ln_var', ln_var.data)

        var_pout = GaussianPolicyOutput(mean, var=var)
        ln_var_pout = GaussianPolicyOutput(mean, ln_var=ln_var)

        np.testing.assert_allclose(var_pout.mean.data, ln_var_pout.mean.data)
        np.testing.assert_allclose(var_pout.var.data, ln_var_pout.var.data)
        np.testing.assert_allclose(
            var_pout.ln_var.data, ln_var_pout.ln_var.data)
        np.testing.assert_allclose(
            var_pout.entropy.data, ln_var_pout.entropy.data)
