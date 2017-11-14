import chainer
import chainer.functions as F
from chainer.initializers import Constant
import chainer.links as L

from chainerrl.initializers import VarianceScalingConstant


class FactorizedNoisyLinear(L.Linear):
    """Linear layer in Factorized Noisy Network

    Args:
        sigma0 (float): scale of initial value of noise-scaling parameters
    """

    def __init__(self, *args, **kwargs):
        sigma0 = kwargs.pop('sigma0', 0.4)
        super(FactorizedNoisyLinear, self).__init__(*args, **kwargs)
        with self.init_scope():
            self.sigma_W = chainer.Parameter(VarianceScalingConstant(sigma0))
            if self.W.data is not None:
                self._initialize_params_sigma(self.W.shape[1])
            self.sigma_b = chainer.Parameter(Constant(sigma0), self.out_size)

    def _eps(self, shape, dtype):
        xp = self.xp
        r = xp.random.standard_normal(shape).astype(dtype)

        # apply the function f
        return xp.copysign(xp.sqrt(xp.abs(r)), r)

    def _initialize_params_sigma(self, in_size):
        self.sigma_W.initialize((self.out_size, in_size))

    def __call__(self, x):
        in_size = x.size // x.shape[0]
        if self.W.data is None:
            self._initialize_params(in_size)
        if self.sigma_W.data is None:
            self._initialize_params_sigma(in_size)

        dtype = x.dtype
        eps_x = self._eps(self.W.shape[1], dtype)
        eps_y = self._eps(self.W.shape[0], dtype)
        noise_W = self.xp.outer(eps_y, eps_x) * self.sigma_W
        noise_b = eps_y * self.sigma_b
        return F.linear(x, self.W + noise_W, self.b + noise_b)
