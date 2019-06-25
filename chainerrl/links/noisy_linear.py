import chainer
from chainer import cuda
import chainer.functions as F
from chainer.initializers import LeCunUniform
import chainer.links as L
import numpy

from chainerrl.functions import muladd
from chainerrl.initializers import VarianceScalingConstant


class FactorizedNoisyLinear(chainer.Chain):
    """Linear layer in Factorized Noisy Network

    Args:
        mu_link (L.Linear): Linear link that computes mean of output.
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, mu_link, sigma_scale=0.4):
        super(FactorizedNoisyLinear, self).__init__()
        self._kernel = None
        self.out_size = mu_link.out_size
        self.nobias = not ('/b' in [name for name, _ in mu_link.namedparams()])

        W_data = mu_link.W.array
        in_size = None if W_data is None else W_data.shape[1]
        device_id = mu_link._device_id

        with self.init_scope():
            self.mu = L.Linear(in_size, self.out_size, self.nobias,
                               initialW=LeCunUniform(1 / numpy.sqrt(3)))

            self.sigma = L.Linear(in_size, self.out_size, self.nobias,
                                  initialW=VarianceScalingConstant(
                                      sigma_scale),
                                  initial_bias=VarianceScalingConstant(
                                      sigma_scale))

        if device_id is not None:
            self.to_gpu(device_id)

    def _noise_function(self, r):
        if self._kernel is None:
            self._kernel = cuda.elementwise(
                '', 'T r',
                '''r = copysignf(sqrtf(fabsf(r)), r);''',
                'noise_func')
        self._kernel(r)

    def _eps(self, shape, dtype):
        xp = self.xp
        if xp is numpy:
            r = xp.random.standard_normal(shape).astype(dtype)
            return xp.copysign(xp.sqrt(xp.abs(r)), r)
        else:
            r = xp.random.standard_normal(shape, dtype)
            self._noise_function(r)
            return r

    def __call__(self, x):
        if self.mu.W.array is None:
            self.mu.W.initialize((self.out_size, numpy.prod(x.shape[1:])))
        if self.sigma.W.array is None:
            self.sigma.W.initialize((self.out_size, numpy.prod(x.shape[1:])))

        # use info of sigma.W to avoid strange error messages
        dtype = self.sigma.W.dtype
        out_size, in_size = self.sigma.W.shape

        eps = self._eps(in_size + out_size, dtype)
        eps_x = eps[:in_size]
        eps_y = eps[in_size:]
        W = muladd(self.sigma.W, self.xp.outer(eps_y, eps_x), self.mu.W)
        if self.nobias:
            return F.linear(x, W)
        else:
            b = muladd(self.sigma.b, eps_y, self.mu.b)
            return F.linear(x, W, b)
