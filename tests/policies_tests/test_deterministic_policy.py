from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import unittest

import chainer
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
import numpy as np

import chainerrl


class _TestDeterministicPolicy(unittest.TestCase):

    def _test_call_given_model(self, model, gpu):
        # This method only check if a given model can receive random input
        # data and return output data with the correct interface.
        min_action = np.full((self.action_size,), -0.01, dtype=np.float32)
        max_action = np.full((self.action_size,), 0.01, dtype=np.float32)
        batch_size = 7
        x = np.random.rand(
            batch_size, self.n_input_channels).astype(np.float32)
        if gpu >= 0:
            model.to_gpu(gpu)
            x = chainer.cuda.to_gpu(x)
            min_action = chainer.cuda.to_gpu(min_action)
            max_action = chainer.cuda.to_gpu(max_action)
        y = model(x)
        self.assertTrue(isinstance(
            y, chainerrl.distribution.ContinuousDeterministicDistribution))
        a = y.sample()
        self.assertTrue(isinstance(a, chainer.Variable))
        self.assertEqual(a.shape, (batch_size, self.action_size))
        self.assertEqual(chainer.cuda.get_array_module(a),
                         chainer.cuda.get_array_module(x))
        if self.bound_action:
            self.assertTrue((a.data <= max_action).all())
            self.assertTrue((a.data >= min_action).all())


@testing.parameterize(
    *testing.product({
        'n_input_channels': [1, 5],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'action_size': [1, 2],
        'bound_action': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCDeterministicPolicy(_TestDeterministicPolicy):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        min_action = np.full((self.action_size,), -0.01, dtype=np.float32)
        max_action = np.full((self.action_size,), 0.01, dtype=np.float32)
        model = chainerrl.policies.FCDeterministicPolicy(
            n_input_channels=self.n_input_channels,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            action_size=self.action_size,
            bound_action=self.bound_action,
            min_action=min_action,
            max_action=max_action,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @attr.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@testing.parameterize(
    *testing.product({
        'n_input_channels': [1, 5],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'action_size': [1, 2],
        'bound_action': [True, False],
        'normalize_input': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCBNDeterministicPolicy(_TestDeterministicPolicy):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        min_action = np.full((self.action_size,), -0.01, dtype=np.float32)
        max_action = np.full((self.action_size,), 0.01, dtype=np.float32)
        model = chainerrl.policies.FCBNDeterministicPolicy(
            n_input_channels=self.n_input_channels,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            action_size=self.action_size,
            bound_action=self.bound_action,
            min_action=min_action,
            max_action=max_action,
            normalize_input=self.normalize_input,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @attr.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@testing.parameterize(
    *testing.product({
        'n_input_channels': [1, 5],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'action_size': [1, 2],
        'bound_action': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCLSTMDeterministicPolicy(_TestDeterministicPolicy):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        min_action = np.full((self.action_size,), -0.01, dtype=np.float32)
        max_action = np.full((self.action_size,), 0.01, dtype=np.float32)
        model = chainerrl.policies.FCLSTMDeterministicPolicy(
            n_input_channels=self.n_input_channels,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            action_size=self.action_size,
            bound_action=self.bound_action,
            min_action=min_action,
            max_action=max_action,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @attr.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)
