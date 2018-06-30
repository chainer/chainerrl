import chainer
import chainer.functions as F
from chainer.initializers import Constant, Uniform
import chainer.links as L
import numpy

from chainerrl.initializers import VarianceScalingConstant


class FactorizedNoisyLinear2(chainer.Chain):
    """Linear layer in Factorized Noisy Network

    Args:
        mu_link (L.Linear): Linear link that computes mean of output.
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, mu_link, sigma_scale=0.4, constant=-1):
        super(FactorizedNoisyLinear2, self).__init__()
        self.out_size = mu_link.out_size
        self.nobias = not ('/b' in [name for name, _ in mu_link.namedparams()])

        W_data = mu_link.W.data
        in_size = None if W_data is None else W_data.shape[1]

        self.device_id = mu_link._device_id

        with self.init_scope():
            self.mu = mu_link
            self.sigma = L.Linear(
                in_size=10, out_size=self.out_size*in_size, nobias=True)

        if self.device_id is not None:
            self.to_gpu(self.device_id)

        self.constant = constant
        self.entropy = None

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

        z = self._eps(10, dtype)
        noise = F.linear(F.reshape(z, (1, -1)), self.sigma.W)
        self.entropy = F.log(noise**2)#F.gaussian_kl_divergence(0, self.xp.log(self.xp.abs(noise)))

        W = self.mu.W + F.reshape(noise, (self.mu.W.shape[0], self.mu.W.shape[1]))
        if self.nobias:
            return F.linear(x, W)
        else:
            b = self.mu.b# + self.sigma.b * eps_y
            return F.linear(x, W, b)
