import chainer
import chainer.functions as F
from chainer.initializers import LeCunUniform
import chainer.links as L
import numpy

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
        self.out_size = mu_link.out_size
        self.nobias = not ('/b' in [name for name, _ in mu_link.namedparams()])

        W_data = mu_link.W.data
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

    def _eps(self, shape, dtype):
        xp = self.xp
        r = xp.random.standard_normal(shape).astype(dtype)

        # apply the function f
        return xp.copysign(xp.sqrt(xp.abs(r)), r)

    def __call__(self, x):
        if self.mu.W.data is None:
            self.mu.W.initialize((self.out_size, numpy.prod(x.shape[1:])))
        if self.sigma.W.data is None:
            self.sigma.W.initialize((self.out_size, numpy.prod(x.shape[1:])))

        # use info of sigma.W to avoid strange error messages
        dtype = self.sigma.W.dtype
        out_size, in_size = self.sigma.W.shape

        eps_x = self._eps(in_size, dtype)
        eps_y = self._eps(out_size, dtype)
        W = self.mu.W + self.sigma.W * self.xp.outer(eps_y, eps_x)
        if self.nobias:
            return F.linear(x, W)
        else:
            b = self.mu.b + self.sigma.b * eps_y
            return F.linear(x, W, b)
