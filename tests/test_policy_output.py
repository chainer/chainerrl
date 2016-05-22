import unittest

import numpy as np
import chainer
import scipy.stats

from policy_output import GaussianPolicyOutput


class TestGaussianPolicyOutput(unittest.TestCase):

    def setUp(self):
        # self.shape = (2, 3)
        self.shape = (1, 1)
        self.mean = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))
        print('mean', self.mean.data)
        print('ln_var', self.mean.data)
        self.ln_var = chainer.Variable(
            np.random.rand(*self.shape).astype(np.float32))

    def test_sampled_actions_log_probs(self):
        pout = GaussianPolicyOutput(self.mean, self.ln_var)
        sampled_actions = pout.sampled_actions
        print('sampled_actions', sampled_actions.data)
        stdev = np.sqrt(np.exp(self.ln_var.data))
        print('stdev', stdev)
        scipy_pdf = scipy.stats.norm(
            self.mean.data, stdev).pdf(sampled_actions.data)
        np.testing.assert_allclose(
            pout.sampled_actions_log_probs.data, np.log(scipy_pdf).sum(axis=1))

    def test_entropy(self):
        pout = GaussianPolicyOutput(self.mean, self.ln_var)
        desired_entropy = 0.5 * \
            np.log(2 * np.pi * np.e * np.exp(self.ln_var.data)).sum(axis=1)
        np.testing.assert_allclose(
            pout.entropy.data, desired_entropy.astype(np.float32))
