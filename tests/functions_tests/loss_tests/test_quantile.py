import unittest

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import numpy as np

from chainerrl import functions
# .loss.quantile import quantile_huber_loss_Dabney
# from chainerrl.functions.loss.quantile import quantile_huber_loss_Dabney


@testing.parameterize(
    {'delta': 0.1},
    {'delta': 1},
    {'delta': 10},
)
class TestQuantileHuberLossDabney(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 10)
        self.x = np.random.standard_normal(self.shape).astype(np.float32)
        self.t = np.random.standard_normal(self.shape).astype(np.float32)
        self.tau = np.random.uniform(0, 1, self.shape).astype(np.float32)
        self.gy = np.random.standard_normal(self.shape).astype(np.float32)

        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
                                       # 'no_grads': [False, False, True]}
                                       # 'no_grads': [True, True, False]}

    def check_forward(self, x_data, t_data, tau_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        tau = chainer.Variable(tau_data)
        loss = functions.quantile_huber_loss_Dabney(x, t, tau, self.delta)
        self.assertEqual(loss.data.dtype, np.float32)
        loss_value = cuda.to_cpu(loss.data)

        coeff = np.where(t_data - x_data > 0, tau_data, 1 - tau_data)
        loss_expect = coeff * cuda.to_cpu(
            F.huber_loss(x, t, self.delta, reduce='no').data)
        testing.assert_allclose(loss_value, loss_expect)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.tau)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                           cuda.to_gpu(self.tau))

    def check_backward(self, x_data, t_data, tau_data, y_grad):
        gradient_check.check_backward(
            functions.QuantileHuberLossDabney(self.delta),
            (x_data, t_data, tau_data), y_grad, eps=1e-2,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.tau, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                            cuda.to_gpu(self.tau), cuda.to_gpu(self.gy))


@testing.parameterize(
    {'delta': 0.1},
    {'delta': 1},
    {'delta': 10},
)
class TestQuantileHuberLossAravkin(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 10)
        self.x = np.random.standard_normal(self.shape).astype(np.float32)
        self.t = np.random.standard_normal(self.shape).astype(np.float32)
        self.tau = np.random.uniform(0, 1, self.shape).astype(np.float32)
        self.gy = np.random.standard_normal(self.shape).astype(np.float32)

        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data, t_data, tau_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        tau = chainer.Variable(tau_data)
        loss = functions.quantile_huber_loss_Aravkin(x, t, tau, self.delta)
        self.assertEqual(loss.data.dtype, np.float32)
        loss_value = cuda.to_cpu(loss.data)

        e = x_data - t_data
        loss_expect = np.where(
            e > self.delta * (1 - tau_data),
            (1 - tau_data) * e - 0.5 * self.delta * (1 - tau_data) ** 2,
            np.where(
                e < - self.delta * tau_data,
                - tau_data * e - 0.5 * self.delta * tau_data ** 2,
                0.5 * e ** 2 / self.delta))
        
        testing.assert_allclose(loss_value, loss_expect)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.tau)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                           cuda.to_gpu(self.tau))

    def check_backward(self, x_data, t_data, tau_data, y_grad):
        gradient_check.check_backward(
            functions.QuantileHuberLossAravkin(self.delta),
            (x_data, t_data, tau_data), y_grad, eps=1e-2,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.tau, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                            cuda.to_gpu(self.tau), cuda.to_gpu(self.gy))
