from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer.functions as F
from chainer import testing
from chainer.testing import attr

import basetest_state_action_q_function as base
import chainerrl


@testing.parameterize(
    *testing.product({
        'n_dim_obs': [1, 5],
        'n_dim_action': [1, 3],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCSAQFunction(base._TestSAQFunction):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = chainerrl.q_functions.FCSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
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
        'n_dim_obs': [1, 5],
        'n_dim_action': [1, 3],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCLSTMSAQFunction(base._TestSAQFunction):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = chainerrl.q_functions.FCLSTMSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
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
        'n_dim_obs': [1, 5],
        'n_dim_action': [1, 3],
        'n_hidden_layers': [0, 1, 2],
        'n_hidden_channels': [1, 2],
        'normalize_input': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCBNSAQFunction(base._TestSAQFunction):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = chainerrl.q_functions.FCBNSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
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
        'n_dim_obs': [1, 5],
        'n_dim_action': [1, 3],
        'n_hidden_layers': [1, 2],  # LateAction requires n_hidden_layers >=1
        'n_hidden_channels': [1, 2],
        'normalize_input': [True, False],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCBNLateActionSAQFunction(base._TestSAQFunction):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = chainerrl.q_functions.FCBNLateActionSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
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
        'n_dim_obs': [1, 5],
        'n_dim_action': [1, 3],
        'n_hidden_layers': [1, 2],  # LateAction requires n_hidden_layers >=1
        'n_hidden_channels': [1, 2],
        'nonlinearity': ['relu', 'elu'],
        'last_wscale': [1, 1e-3],
    })
)
class TestFCLateActionSAQFunction(base._TestSAQFunction):

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = chainerrl.q_functions.FCLateActionSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @attr.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)
