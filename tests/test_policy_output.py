import unittest

import numpy as np
import chainer
import scipy.stats

from policy_output import GaussianPolicyOutput


class TestGaussianPolicyOutput(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 3)
        self.mean = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('mean', self.mean.data)
        self.ln_var = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('ln_var', self.ln_var.data)

    def test_sampled_actions_log_probs(self):
        pout = GaussianPolicyOutput(self.mean, self.ln_var)
        sampled_actions = pout.sampled_actions
        print('sampled_actions', sampled_actions.data)
        for i in range(self.shape[0]):
            cov = np.identity(
                self.shape[1], dtype=np.float32) * np.exp(self.ln_var.data[i])
            print('cov', cov)
            desired_pdf = scipy.stats.multivariate_normal(
                self.mean.data[i], cov).pdf(sampled_actions.data[i])
            np.testing.assert_allclose(
                pout.sampled_actions_log_probs.data[i], np.log(desired_pdf))

    def test_entropy(self):
        pout = GaussianPolicyOutput(self.mean, self.ln_var)
        for i in range(self.shape[0]):
            cov = np.identity(
                self.shape[1], dtype=np.float32) * np.exp(self.ln_var.data[i])
            print('cov', cov)
            desired_entropy = scipy.stats.multivariate_normal(
                self.mean.data[i], cov).entropy()
            np.testing.assert_allclose(
                pout.entropy.data[i], desired_entropy.astype(np.float32))
