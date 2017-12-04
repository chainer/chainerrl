from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer.utils import type_check
import numpy


def quantile_loss(y, t, tau):
    e = t - y
    return F.maximum(
        e * tau,
        e * (tau - 1.)
    )


def quantile_huber_loss_Aravkin(y, t, tau, delta=1.):
    """Quantile Huber loss a la Aravkin+

    See: https://arxiv.org/abs/1402.4624
    """
    assert False
    e = t - y
    # if e in the interval [- delta * tau, delta * (1 - tau)]
    e ** 2 / (2 * delta)
    # else
    F.maximum(
        e * tau - delta * tau**2 / 2,
        e * (tau - 1.) - delta * (tau - 1.)**2 / 2
    )


class QuantileHuberLossDabney(function.Function):

    def __init__(self, delta):
        self.delta = delta

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[2].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape == in_types[2].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t, tau = inputs
        self.diff = t - x
        l1 = xp.abs(self.diff)
        self.mask = l1 > self.delta

        sign = xp.sign(self.diff)
        # self.coeff = xp.where(self.diff > 0, tau, 1 - tau)
        self.coeff = 0.5 + sign * (tau - 0.5)

        l2 = xp.square(self.diff)
        self.huber = 0.5 * (l2 - self.mask * xp.square(l1 - self.delta))
        return self.huber * self.coeff,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)

        sign = xp.sign(self.diff)
        gd = xp.where(self.mask, sign * self.delta, self.diff)
        gd *= self.coeff
        gy_ = gy[0]
        gd = gy_ * gd
        gtau = gy_ * sign * self.huber
        return -gd, gd, gtau


def quantile_huber_loss_Dabney(y, t, tau, delta=1.):
    """Quantile Huber loss a la Dabney+

    See: https://arxiv.org/abs/1710.10044
    """
    return QuantileHuberLossDabney(delta)(y, t, tau)
