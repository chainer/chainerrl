import chainer
import chainer.functions as F
from chainer.initializers import Constant, Uniform
from chainer.initializers import LeCunUniform
import chainer.links as L
import numpy

from chainerrl.initializers import VarianceScalingConstant

from chainer import initializer
class VarianceScalingUniform(initializer.Initializer):
    def __init__(self, scale=1.0, dtype=None):
        super(VarianceScalingUniform, self).__init__(dtype)
        self.scale = scale

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype

        if len(array.shape) == 1:
            Uniform(self.scale / numpy.sqrt(array.shape[0]))(array)
        else:
            fan_in, _ = initializer.get_fans(array.shape)
            Uniform(self.scale / numpy.sqrt(fan_in))(array)

PARA = True

class FactorizedNoisyLinear(chainer.Chain):
    """Linear layer in Factorized Noisy Network

    Args:
        mu_link (L.Linear): Linear link that computes mean of output.
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, mu_link, sigma_scale=0.4, constant=-1, prev=False, noise_coef=1,
        init_method='/out'):
        super(FactorizedNoisyLinear, self).__init__()
        self.out_size = mu_link.out_size
        self.nobias = not ('/b' in [name for name, _ in mu_link.namedparams()])
        self.entropy = 0
        self.off = False
        self.noise_coef = noise_coef

        W_data = mu_link.W.data
        in_size = None if W_data is None else W_data.shape[1]

        self.device_id = mu_link._device_id

        with self.init_scope():
            self.mu = L.Linear(in_size, self.out_size, self.nobias,
                               initialW=LeCunUniform(1 / numpy.sqrt(3)))

            self.sigma = L.Linear(in_size, self.out_size, self.nobias,
                                  initialW=VarianceScalingConstant(
                                      sigma_scale),
                                  initial_bias=VarianceScalingConstant(
                                      sigma_scale))

        if self.device_id is not None:
            self.to_gpu(self.device_id)

        self.constant = constant

        self.reset_noise()

    def _eps(self, shape, dtype):
        xp = self.xp
        r = xp.random.standard_normal(shape).astype(dtype)

        # apply the function f
        return xp.copysign(xp.sqrt(xp.abs(r)), r)

    def reset_noise(self):
        dtype = self.sigma.W.dtype
        out_size, in_size = self.sigma.W.shape

        if PARA:
            self.eps_x = self._eps(in_size, dtype)
            self.eps_y = self._eps(out_size, dtype)

    def __call__(self, x, noise=False, act=False):
        if self.mu.W.data is None:
            self.mu.W.initialize((self.out_size, numpy.prod(x.shape[1:])))
        if self.mu.b.data is None:
            self.mu.b.initialize((self.out_size,))
        if self.sigma.W.data is None:
            self.sigma.W.initialize((self.out_size, numpy.prod(x.shape[1:])))

        if not PARA:
            noise = True
            dtype = self.sigma.W.dtype
            out_size, in_size = self.sigma.W.shape
            self.eps_x = self._eps(in_size, dtype)
            self.eps_y = self._eps(out_size, dtype)

        if noise and not act:
            if PARA:
                W = self.mu.W + self.xp.outer(self.eps_y, self.eps_x) * self.noise_coef
            else:
                W = self.mu.W + self.sigma.W * self.xp.outer(self.eps_y, self.eps_x)# * self.noise_coef
        else:
            W = self.mu.W

        if self.nobias:
            # gaussian entropy: 0.5 * log(2*pi*e*(sigma**2))
            self.entropy = F.sum(F.log(self.sigma.W**2+1e-5))
            return F.linear(x, W)
        else:
            self.entropy = F.sum(F.log(self.sigma.W**2+1e-5)) + F.sum(F.log(self.sigma.b**2+1e-5))
            if noise and not act:
                if PARA:
                    b = self.mu.b + self.eps_y * self.noise_coef
                else:
                    b = self.mu.b + self.sigma.b * self.eps_y# * self.noise_coef
            else:
                b = self.mu.b
            return F.linear(x, W, b)
